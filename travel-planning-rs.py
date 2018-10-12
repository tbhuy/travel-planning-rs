import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import hdbscan
from haversine import haversine
import itertools
from sklearn.cluster import KMeans # Importing KMeans

url = 'http://dbpedia.org/sparql/'
c="""
select distinct ?countrya where {?countrya skos:broader dbc:Visitor_attractions_by_country_and_city}  order by ?countrya
"""
r = requests.get(url, params = {'format': 'json', 'query': c})
data = r.json()
country = []
i = 1
#print(data)
for item in data['results']['bindings']:
    countryname = item['countrya']['value']
    if countryname.find("_in_")!=-1:
        print(i, countryname[countryname.find("_in_")+4:countryname.find("_by_")])
        country.append(item['countrya']['value'])
        i = i + 1
    
selectedcountry = int(input("Select a country:"))
#print(country[selectedcountry-1])

c="""
SELECT distinct ?citya WHERE {
 
 ?citya skos:broader ?countrya.
filter(?countrya=<"""+country[selectedcountry-1]+""">)

}
"""
#print(c)
r = requests.get(url, params = {'format': 'json', 'query': c})
data = r.json()
city = []
i = 1
#print(data)
for item in data['results']['bindings']:
    cityname = item['citya']['value']
    print(i, cityname[cityname.find("_in_")+4:])
    city.append(item['citya']['value'])
    i = i + 1
    
selectedcity = int(input("Select a city:"))

query = """
select * where {
?place rdf:type dbo:Place.
?place geo:lat ?lat.
?place geo:long ?long.
?place rdfs:label ?label.
?place dct:subject ?subj.
 FILTER (lang(?label) = 'en' and ?subj=<"""+city[selectedcity-1]+""">)
}
"""
#print(query)
cityname = city[selectedcity-1]
title = cityname[cityname.find("_in_")+4:]
r = requests.get(url, params = {'format': 'json', 'query': query})
data = r.json()
#print(data)
coord_col = ['Longitude', 'Latitude', 'Landmark']
df_coord = pd.DataFrame(columns=coord_col) 
for item in data['results']['bindings']:
     df_coord = df_coord.append({'Longitude': float(item['long']['value']),'Latitude': float(item['lat']['value']), 'Landmark':item['label']['value']},ignore_index=True)
print("There are",len(df_coord),"attracttions in", title) 
ncluster = int(input("Number of days to stay?"))
#print(df_coord)
X = df_coord[['Longitude','Latitude']].values
#rads = np.radians(X)
#clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='haversine')
#predictions = clusterer.fit_predict(rads)
kmeans = KMeans(n_clusters=ncluster)
predictions = kmeans.fit_predict(X)
clustered = pd.concat([df_coord.reset_index(), 
                       pd.DataFrame({'Cluster':predictions})], 
                      axis=1)
clustered.drop('index', axis=1, inplace=True)
#print(clustered)                    
fig = plt.figure(figsize=(16,8))
cmap=plt.cm.rainbow
norm = matplotlib.colors.BoundaryNorm(np.arange(0,ncluster+1,1), cmap.N)
plt.scatter(clustered['Longitude'], clustered['Latitude'], c=clustered['Cluster'],
            cmap=cmap, norm=norm, s=150, edgecolor='none')
#plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
#centers = kmeans.cluster_centers_
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.3);
plt.xlabel('Latitude', fontsize=14)
plt.ylabel('Longitude', fontsize=14)
plt.grid()

plt.title(str(ncluster)+" days in "+title, fontsize=14)

for item in df_coord[['Longitude','Latitude','Landmark']].values:
    plt.annotate(item[2], (item[0], item[1]))

for i in range(0,kmeans.labels_.max()+1):
    cluster = clustered.loc[clustered['Cluster']==i]
    cluster.reset_index(inplace=True, drop=True)
    print(cluster)
    if len(cluster) >= 2:
        init = range(0,len(cluster))
        dist = []
        path =  list(itertools.permutations(init))
        #print(path)
        for x in path:
            dis = 0
            for y in range(0,len(x)-1):           
               dis = dis +  haversine( (cluster.loc[x[y],'Latitude'],cluster.loc[x[y],'Longitude']),  (cluster.loc[x[y + 1],'Latitude'],cluster.loc[x[y + 1],'Longitude']))
            dist.append(dis)
        print("Shortest itinerary: ", path[dist.index(min(dist))], "covering a distance of", round(min(dist),3), " km")
        spath = path[dist.index(min(dist))]
        for j in range(0,len(spath)-1):
            plt.plot((cluster.loc[spath[j],'Longitude'],cluster.loc[spath[j+1],'Longitude']),  (cluster.loc[spath[j],'Latitude'],cluster.loc[spath[j+1],'Latitude']) , color='r',linewidth=2)
            dis =  round(haversine( (cluster.loc[spath[j],'Latitude'],cluster.loc[spath[j],'Longitude']),  (cluster.loc[spath[j+1],'Latitude'],cluster.loc[spath[j+1],'Longitude'])),2)
            plt.annotate(dis, ((cluster.loc[spath[j],'Longitude']+cluster.loc[spath[j+1],'Longitude'])/2, (cluster.loc[spath[j],'Latitude']+cluster.loc[spath[j+1],'Latitude'])/2))
         
         
plt.show() 
