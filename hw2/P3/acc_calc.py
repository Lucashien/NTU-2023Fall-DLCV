import csv

test_name = []
test_label = []
with open("hw2_data/digits/svhn/test.csv", "r", newline="") as file:
    reader = csv.reader(file)
    next(reader, None)
    for row in reader:
        name = row[0]
        label = row[1]
        test_name.append(name)
        test_label.append(label)

length = 0
correct = 0
with open("hw2_data/digits/svhn/val.csv", "r", newline="") as file:
    reader = csv.reader(file)
    next(reader, None)
    for row in reader:
        val_name = row[0]
        val_label = row[1]
        if val_name in test_name:
            length+=1
            if  val_label==test_label[test_name.index(val_name)]:
               correct +=1
               
print(f"acc:{100*(correct/length):3f}%")
        
