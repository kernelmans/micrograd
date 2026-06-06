from micrograd.engine import Value

# Second anatomical experiment: a single artificial neuron.
# We observe the chain rule through multiplication, addition, and ReLU.

# Inputs / parameters
x = Value(2.0)
w = Value(-3.0)
b = Value(10.0)

# Forward pass: n = w*x + b
n = w * x + b

# Non-linearity: y = ReLU(n)
y = n.relu()

# Backward pass: compute dy/dx, dy/dw, dy/db
y.backward()

print("Forward:")
print("x =", x)
print("w =", w)
print("b =", b)
print("n = w*x + b =", n)
print("y = relu(n) =", y)

print("\nGradients:")
print("dy/dx =", x.grad)
print("dy/dw =", w.grad)
print("dy/db =", b.grad)
