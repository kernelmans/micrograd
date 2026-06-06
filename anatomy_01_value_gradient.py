from micrograd.engine import Value

a = Value(2.0)
b = Value(3.0)

c = a * b
d = c + 1

d.backward()

print("Forward:")
print("c = a * b =", c.data)
print("d = c + 1 =", d.data)

print("\nBackward:")
print("d/da =", a.grad)
print("d/db =", b.grad)