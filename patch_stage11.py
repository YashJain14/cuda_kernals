with open("stage11.cu", "r") as f:
    content = f.read()

# Fix torch scalar initialization
content = content.replace(
"""constexpr FlashForwardKernelConfig cfg_stage11 = {
    torch::kFloat16, 64, 128, 64, 4,
    true, true, true,
    0, 0, 0,
    true, true
};""",
"""constexpr FlashForwardKernelConfig cfg_stage11 = {
    1, 64, 128, 64, 4,
    true, true, true,
    0, 0, 0,
    true, true
};""")

with open("stage11.cu", "w") as f:
    f.write(content)
