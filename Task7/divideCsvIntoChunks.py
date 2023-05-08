import csv
import time
with open('india-news-headlines.csv', 'r') as file:

    csv_reader = csv.reader(file)
    headers = next(csv_reader)
    i = 1

    while True:
        isFileEnd = False
        with open(f'india-news-headlines-{i}.csv', 'w') as file:
            file.write(','.join(headers) + '\n')
            for j, row in enumerate(csv_reader):
                file.write(','.join(row) + '\n')
                if j > 500000:
                    break
            else:
                isFileEnd = True
            print(f'File Created{i}')
            if isFileEnd:
                break

        i += 1

print("Finished")
        
    



