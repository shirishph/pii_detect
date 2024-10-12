from faker import Faker

TOTAL_TRAINING_DATAPOINTS = 20000
TOTAL_HOLDOUT_DATAPOINTS = 1000

fake = Faker()

def create_fake_data(key):
  if key == "creditcard":
    return fake.credit_card_number()
  if key == "datetime":
    return str(fake.date_time_this_decade())
  if key == "emailaddress":
    return fake.ascii_company_email()
  if key == "ibancode":
    return fake.iban()
  if key == "ipaddress":
    return fake.ipv4()
  if key == "nrp":
    return fake.country() # Todo: Tech debt, this is nation and not nationality
  if key == "location":
    return fake.address().replace("\n", ", ")
  if key == "person":
    return fake.name()
  if key == "phonenumber":
    return fake.phone_number()
  if key == "url":
    return fake.uri() # Todo: Tech debt, expectation is of URI but has a misnomer of URL
  if key == "usabanknumber":
    return fake.aba() # Todo: Use a proper bank number 8-17 digits
  if key == "usadriverlicense":
    return fake.bothify(text='?#######') # CA = ANNNNNNN, Todo: Extend for all relevant nations & states
  if key == "usaitin":
    return fake.itin()
  if key == "usapassport":
    return fake.passport_number()
  if key == "usassn":
    return fake.ssn()

  return "Error"

keys = [
  "creditcard",
  "datetime",
  "emailaddress",
  "ibancode",
  "ipaddress",
  "nrp",
  "location",
  "person",
  "phonenumber",
  "url",
  "usabanknumber",
  "usadriverlicense",
  "usaitin",
  "usapassport",
  "usassn"
]

"""
for key in keys:
  print(key + ": " + create_fake_data(key))
"""

datasets = [
  (TOTAL_TRAINING_DATAPOINTS, "train-dataset.tsv"),
  (TOTAL_HOLDOUT_DATAPOINTS, "holdout-dataset.tsv")
]

for dataset in datasets:
  with open(dataset[1], "w") as fw:
    fw.write("data\tlabels\n")
    key_counter = 0
    for i in range(dataset[0]):
      fw.write(create_fake_data(keys[key_counter]) + "\t" + keys[key_counter] + "\n")
      key_counter += 1
      if key_counter > len(keys) - 1:
        # fw.write("------------------------------------------------------\n")
        key_counter = 0
