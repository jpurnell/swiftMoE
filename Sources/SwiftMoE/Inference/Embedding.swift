import Foundation

/// Embedding table lookup and lm_head projection.
///
/// Both operations are 4-bit dequantized matrix-vector multiplies against the
/// mmap'd weight file. The embedding is a row lookup (one row per token),
/// while lm_head is a full matvec (4096 → 248,320).
public enum Embedding {

    /// Looks up a token's embedding from the quantized embedding table.
    ///
    /// Dequantizes a single row of the embedding matrix:
    /// `[vocab_size=248320, hidden_dim=4096]` stored as 4-bit packed uint32.
    ///
    /// Matches `embed_lookup` in `infer.m:2863-2908`.
    ///
    /// - Parameters:
    ///   - weightFile: The mmap'd weight file.
    ///   - tokenID: Token index (0..<248320).
    ///   - config: Model configuration providing dimensions and group size.
    ///   - output: Destination buffer [HIDDEN_DIM floats].
    public static func lookup(
        weightFile: WeightFile,
        tokenID: Int,
        config: ModelConfig,
        output: UnsafeMutablePointer<Float>
    ) {
        let hiddenDim = config.hiddenDim   // 4096
        let groupSize = config.groupSize    // 64
        let numGroups = hiddenDim / groupSize    // 64
        let packedCols = hiddenDim / 8           // 512

        guard let W = weightFile.tensorPointer(name: "model.embed_tokens.weight", as: UInt32.self),
              let S = weightFile.tensorPointer(name: "model.embed_tokens.scales", as: UInt16.self),
              let B = weightFile.tensorPointer(name: "model.embed_tokens.biases", as: UInt16.self) else {
            // Zero output if embedding not found
            memset(output, 0, hiddenDim * MemoryLayout<Float>.size)
            return
        }

        let wRow = W + tokenID * packedCols
        let sRow = S + tokenID * numGroups
        let bRow = B + tokenID * numGroups

        for g in 0..<numGroups {
            let scale = bf16ToFloat(sRow[g])
            let bias = bf16ToFloat(bRow[g])
            let packedPerGroup = groupSize / 8  // 8

            for p in 0..<packedPerGroup {
                let packed = wRow[g * packedPerGroup + p]
                let baseIdx = g * groupSize + p * 8

                for n in 0..<8 {
                    let nibble = (packed >> (n * 4)) & 0xF
                    output[baseIdx + n] = Float(nibble) * scale + bias
                }
            }
        }
    }

    /// Projects hidden state to logits via the language model head.
    ///
    /// Computes: `logits[248320] = lm_head_weight[248320, 4096] @ hidden[4096]`
    /// using 4-bit dequantized matrix-vector multiply.
    ///
    /// Matches `lm_head_forward` in `infer.m:2914-2934`.
    ///
    /// - Parameters:
    ///   - weightFile: The mmap'd weight file.
    ///   - hidden: Input hidden state [HIDDEN_DIM floats].
    ///   - config: Model configuration providing vocab size, hidden dim, and group size.
    ///   - logits: Output logits [VOCAB_SIZE floats].
    public static func lmHead(
        weightFile: WeightFile,
        hidden: UnsafePointer<Float>,
        config: ModelConfig,
        logits: UnsafeMutablePointer<Float>
    ) {
        let vocabSize = config.vocabSize
        let hiddenDim = config.hiddenDim
        let groupSize = config.groupSize

        guard let W = weightFile.tensorPointer(name: "lm_head.weight", as: UInt32.self),
              let S = weightFile.tensorPointer(name: "lm_head.scales", as: UInt16.self),
              let B = weightFile.tensorPointer(name: "lm_head.biases", as: UInt16.self) else {
            memset(logits, 0, vocabSize * MemoryLayout<Float>.size)
            return
        }

        // CPU dequantized matvec: logits = W_4bit @ hidden
        cpuDequantMatvec(
            W: W, scales: S, biases: B,
            input: hidden, output: logits,
            outDim: vocabSize, inDim: hiddenDim, groupSize: groupSize
        )
    }

    /// Greedy argmax sampling — returns the token ID with the highest logit.
    ///
    /// Matches `cpu_argmax` in `infer.m:843-851`.
    public static func argmax(logits: UnsafePointer<Float>, vocabSize: Int) -> Int {
        var bestIdx = 0
        var bestVal = logits[0]
        for i in 1..<vocabSize {
            if logits[i] > bestVal {
                bestVal = logits[i]
                bestIdx = i
            }
        }
        return bestIdx
    }

    // MARK: - CPU Dequantized Matvec

    /// CPU reference 4-bit dequantized matrix-vector multiply.
    ///
    /// Matches `cpu_dequant_matvec` in `infer.m:108-148`.
    static func cpuDequantMatvec(
        W: UnsafePointer<UInt32>,
        scales: UnsafePointer<UInt16>,
        biases: UnsafePointer<UInt16>,
        input: UnsafePointer<Float>,
        output: UnsafeMutablePointer<Float>,
        outDim: Int,
        inDim: Int,
        groupSize: Int
    ) {
        let numGroups = inDim / groupSize
        let packedPerGroup = groupSize / 8
        let packedCols = inDim / 8

        for row in 0..<outDim {
            var acc: Float = 0
            let wRow = W + row * packedCols
            let sRow = scales + row * numGroups
            let bRow = biases + row * numGroups

            for g in 0..<numGroups {
                let scale = bf16ToFloat(sRow[g])
                let bias = bf16ToFloat(bRow[g])
                let basePacked = g * packedPerGroup
                let baseX = g * groupSize

                for p in 0..<packedPerGroup {
                    let packed = wRow[basePacked + p]
                    let xBase = baseX + p * 8

                    for n in 0..<8 {
                        let nibble = (packed >> (n * 4)) & 0xF
                        let wVal = Float(nibble) * scale + bias
                        acc += wVal * input[xBase + n]
                    }
                }
            }
            output[row] = acc
        }
    }
}
