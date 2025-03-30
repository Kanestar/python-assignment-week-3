def calculate_discount(price, discount_percent):
    """
    Calculate the final price after applying a discount.

    Args:
        price (float): The original price of the item.
        discount_percent (float): The discount percentage.

    Returns:
        float: The final price after applying the discount, or the original price if the discount is less than 20%.
    """
    if discount_percent >= 20:
        discount_amount = price * (discount_percent / 100)
        return price - discount_amount
    else:
        return price

def main():
    # Prompt the user to enter the original price and discount percentage
    price = float(input("Enter the original price: $"))
    discount_percent = float(input("Enter the discount percentage (%): "))

    # Calculate the final price after applying the discount
    final_price = calculate_discount(price, discount_percent)

    # Print the final price
    if discount_percent >= 20:
        print(f"Discount applied: {discount_percent}%")
        print(f"Original price: ${price:.2f}")
        print(f"Discount amount: ${price * (discount_percent / 100):.2f}")
        print(f"Final price: ${final_price:.2f}")
    else:
        print(f"No discount applied (discount is less than 20%)")
        print(f"Original price: ${price:.2f}")

if __name__ == "__main__":
    main()