import csv
import time

def submit_feedback(result_id, rating):
    # Append feedback to a CSV file
    with open('feedback.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([result_id, rating, time.strftime("%Y-%m-%d %H:%M:%S")])  # Add timestamp
