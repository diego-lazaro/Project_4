import pickle

with open('Resources/Trained_Models/oldStandard.pkl', 'rb') as f:
    oldStandard = pickle.load(f)

with open('Resources/Trained_Models/current.pkl', 'rb') as f:
    current = pickle.load(f)

with open('Resources/Trained_Models/under5.pkl', 'rb') as f:
    under5 = pickle.load(f)

#Lat 24.5 to 50.0
#Lon -125.0 to -67.0

lats = (5000-2400)
lons = (12500-6700)
coords = []


for x in range(lats):
  lat = ((x/100)+24.0)
  for y in range(lons):
     lon = (((y/100)+67.0)*-1)
     coords.append([lat, lon])

over10 = under5.predict(coords)
over10coords = []
under10coords = []

for x in range((len(over10)-1)):
    if (over10[x] == 1):
       over10coords.append(coords[x])
    if (over10[x] == 0):
       under10coords.append(coords[x])
    

print(over10coords)
   
# over10 = current.predict()
