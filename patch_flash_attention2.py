with open("stage12_include/flash_attention.cuh", "r") as f:
    content = f.read()

content = content.replace("const torch::ScalarType dtype;", "const int dtype;")

with open("stage12_include/flash_attention.cuh", "w") as f:
    f.write(content)
