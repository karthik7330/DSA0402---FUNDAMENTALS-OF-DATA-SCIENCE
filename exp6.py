prices = [50, 100, 80]
quantities = [2, 1, 3]

discount_rate = 10
tax_rate = 5

subtotal = 0

for i in range(len(prices)):
    subtotal = subtotal + (prices[i] * quantities[i])

discount = (subtotal * discount_rate) / 100

amount_after_discount = subtotal - discount

tax = (amount_after_discount * tax_rate) / 100

total_cost = amount_after_discount + tax

print("Subtotal:", subtotal)
print("Discount:", discount)
print("Tax:", tax)
print("Total Cost:", total_cost)