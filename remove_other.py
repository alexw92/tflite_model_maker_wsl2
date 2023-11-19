import sys
import os

def remove_lines_with_infix(input_filename, infix_to_remove):
    output_filename = os.path.splitext(input_filename)[0] + "_no_" + infix_to_remove + os.path.splitext(input_filename)[1]

    with open(input_filename, 'r') as file:
        lines = file.readlines()

    # Remove lines containing the infix
    filtered_lines = [line for line in lines if infix_to_remove not in line]

    with open(output_filename, 'w') as file:
        file.writelines(filtered_lines)

    print(f"Lines containing '{infix_to_remove}' removed. Output written to '{output_filename}'.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_filename infix_to_remove")
    else:
        input_filename = sys.argv[1]
        infix_to_remove = sys.argv[2]

        remove_lines_with_infix(input_filename, infix_to_remove)
