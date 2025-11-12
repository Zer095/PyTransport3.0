import os
import csv
import numpy as np

# Define the subfolder containing the .out files
subfolder = 'SqData'

numbers = range(5000)
sq_num = []

# Check if the subfolder exists
if not os.path.exists(subfolder):
    print(f"Error: The folder '{subfolder}' does not exist.")
else:
    # List to store all rows
    all_rows = []

    
    # Process each .out file in the subfolder
    for filename in os.listdir(subfolder):
        if filename.endswith('.out'):  # Only process .out files
            filepath = os.path.join(subfolder, filename)

            try:
                num = int( filename.split('.')[0].replace('z','') )
                sq_num.append(num)
            except ValueError:
                print(f"Error, str = {filename.split('.')[0].replace('z','')}")
            
            try:
                # Open and read the .out file
                with open(filepath, 'r') as r:
                    lines = r.readlines()
                    
                    # Extract values from the lines
                    fact = float(lines[0].strip())
                    k = float(lines[1].strip())
                    pzt = float(lines[2].strip())
                    pzm = float(lines[3].strip())
                    fnlt = float(lines[4].strip())
                    fnlm = float(lines[5].strip())
                    
                    # Add the row to the list
                    all_rows.append([fact, k, pzt, pzm, fnlt, fnlm])
            
            except FileNotFoundError:
                print(f"File not found: {filepath}")
            except ValueError:
                print(f"Error parsing data in file: {filepath}")
            except IndexError:
                print(f"File format issue (not enough lines) in: {filepath}")

    for n in numbers:
        if n not in sq_num:
            print(f'File number {n} does NOT exist')

    # Sort rows by the first column (`fact`)
    all_rows.sort(key=lambda row: row[0])  # Sorts in ascending order of `fact`

    # Write the sorted data to the CSV file
    with open('SqTotal.csv', 'w', newline='') as f:
        csvwriter = csv.writer(f)
        
        # Write headers
        csvwriter.writerow(['fact', 'k_s', 'pzt', 'pzm', 'fnlt', 'fnlm'])
        
        # Write sorted rows
        for row in all_rows:
            # Format the values for scientific notation
            formatted_row = [f"{value:.18e}" for value in row]
            csvwriter.writerow(formatted_row)
    
    print("Data successfully written to SqTotal.csv in sorted order.")
