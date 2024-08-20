##############
import cartopy.crs as ccrs
import cartopy
import cartopy.mpl.geoaxes
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
import shapefile
import json

####################################


with open(r'/silod5/rummel/runoff_pred_paper/rivers.json') as f:
    coordinates = json.load(f)

boundaries = cartopy.feature.NaturalEarthFeature(category='cultural', facecolor='None',edgecolor='k', name='admin_0_boundary_lines_land', scale='10m')
land = cartopy.feature.NaturalEarthFeature('physical', 'land',scale='10m', edgecolor='k', facecolor=cfeature.COLORS['land'])
ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', scale='10m', edgecolor='none', facecolor=cfeature.COLORS['water'])
lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes', scale='10m', edgecolor=cfeature.COLORS['water'], facecolor=cfeature.COLORS['water'])
rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines',
            scale='10m', edgecolor=cfeature.COLORS['water'], facecolor='none')


plt.figure(figsize=(14, 12))
ax = plt.axes(projection=cartopy.crs.PlateCarree())
ax.add_feature(land)
ax.add_feature(ocean)									
ax.add_feature(rivers, linewidth=0.5, edgecolor=cfeature.COLORS['water'])
ax.add_feature(lakes, alpha=1, linewidth=0.5)
ax.add_feature(boundaries, linewidth=0.5, edgecolor='gray')
ax.set_facecolor(cfeature.COLORS['water'])
ax.coastlines('10m', linewidth=0.1)
ax.set_extent([3, 31, 51, 68])
ax.set_aspect("1.8", adjustable='datalim')

ax.set_xticks(ticks=[3,6,9, 12, 15, 18,21, 24, 27, 30], crs=ccrs.PlateCarree())
ax.set_yticks(ticks=[52, 54, 56, 58, 60, 62, 64, 66], crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda lon, _: f'{lon:.0f}°E'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda lat, _: f'{lat:.0f}°N'))
ax.tick_params(axis='both', which='major', labelsize=18) 

ax.text(29.3, 59.45, 'Neva', fontsize=18,  color='red', transform=ccrs.PlateCarree())
ax.text(14.6, 53.5, 'Oder', fontsize=18,  color='red', transform=ccrs.PlateCarree())
ax.text(21.16, 55.2, 'Neman', fontsize=18,  color='red', transform=ccrs.PlateCarree())
ax.text(18.0, 63.7, 'Umeaelven', fontsize=18,  color='red', transform=ccrs.PlateCarree())

for key, locations in coordinates.items():
    for location in locations:
        lon = location['lon']
        lat = location['lat']
        ax.scatter(lon, lat, color='red',  s=40, alpha=0.8, transform=ccrs.PlateCarree(), zorder=2)
        
ax.scatter(20.04, 57.31, color='red', marker='x', s=50, alpha=0.8, transform=ccrs.PlateCarree(), zorder=3)
ax.text(19.8, 57.5, 'BY15', fontsize=18, color='red', transform=ccrs.PlateCarree(), zorder=3)        

ax.text(4, 56, 'North\nSea', fontsize=22, fontstyle='italic', color='royalblue')
ax.text(20, 58.5, 'Baltic\nSea', fontsize=22, fontstyle='italic', color='royalblue')
ax.text(8, 57.8, 'Skagerrak', fontsize=18,fontstyle='italic', color='royalblue', transform=ccrs.PlateCarree())


plt.savefig(r'/silod5/rummel/runoff_pred_paper/Baltic_Sea_Map', dpi=300, bbox_inches='tight')







