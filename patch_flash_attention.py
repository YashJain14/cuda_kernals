with open("stage11_include/flash_attention.cuh", "r") as f:
    content = f.read()

# Make args not const
content = content.replace("    const index_t batch_stride;", "    index_t batch_stride;")
content = content.replace("    const index_t seq_stride;", "    index_t seq_stride;")
content = content.replace("    const index_t head_stride;", "    index_t head_stride;")
content = content.replace("    const index_t seq_len;", "    index_t seq_len;")
content = content.replace("    const index_t n_heads;", "    index_t n_heads;")
content = content.replace("    const int n_Q_blocks;", "    int n_Q_blocks;")
content = content.replace("    const int n_KV_blocks;", "    int n_KV_blocks;")

with open("stage11_include/flash_attention.cuh", "w") as f:
    f.write(content)
