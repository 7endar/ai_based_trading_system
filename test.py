import torch

if torch.cuda.is_available():
    print("GPU mevcut ve kullanılabilir!")
else:
    print("GPU kullanılamıyor, CPU üzerinde çalışacak.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")