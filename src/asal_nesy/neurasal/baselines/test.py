import torch

range_tensor = (torch.arange(3), torch.arange(3))
z = torch.zeros(2, 3, 4, 4)
y = torch.rand(2, 5, 2, 1, 5, 5)
print(range_tensor)
print(y)
e = torch.eye(3)
print(e)
x = torch.randn(4, 5, device="cuda", dtype=torch.float16)
print(x)
print(y.ndim)

A = torch.randn(3, 3)
b = torch.randn(3)

print(A)
print(b)

x = torch.randn(1000)
lse = torch.logsumexp(x, dim=0)
print(x)
print(lse)

logits = torch.randn(5, 5)
print(logits)
probs = torch.softmax(logits, dim=-1)
print(probs)
probs = torch.softmax(logits, dim=0)
print(probs)

labels = torch.randint(0, 5, (5,))
print(labels)
loss = torch.nn.functional.cross_entropy(logits, labels)
print(loss)

x = torch.randn(10)
y = torch.where(x > 0, x, 777)
print(y)

x = torch.tensor([1,2,3]); y = torch.tensor([4, 5, 6])
print (x * y)

x = torch.tensor([[1,2,3], [5, 6, 7]]); y = torch.tensor([[4, 5, 6], [1,2, 3]])
print (x * y)
print(x.sum(dim=-1))

x = torch.tensor([[1,2,3], [5,6,7]])
y = torch.tensor([[1,2], [5,6], [3,4]])
print(x @ y)

f = lambda u: (u**2).sum()
u = torch.ones(5, 3, 3, requires_grad=True)
print(u)
print(f(u))
g = torch.autograd.grad(f(u), u)[0] # gradient ∂f/∂u = 2u
print(g)

# Dot product
a = torch.randn(5); b = torch.randn(5)
dotprod = a @ b
print(dotprod)
einsum = torch.einsum('i,i->', a, b)
print(einsum)