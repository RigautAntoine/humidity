import pandas as pd
import numpy as np
import os
import xarray as xr
from cartopy import crs
import panel as pn
import bokeh as bk
import hvplot.pandas
import hvplot.xarray
import holoviews as hv

##### EDIT #####
import datashader as dsh
from holoviews.operation.datashader import regrid
import geoviews as gv
#gv.extension('bokeh')


# -------------------------- Chart Builders ---------------------------- #
selector_elem_size = 20

def build_chart(tbl):
    
    if tbl.shape[1] == 1:
        ndplot = hv.NdOverlay({tbl.columns[0]: tbl.hvplot()})
    else:
        ndplot = tbl.hvplot()
    
    ndplot['Safety Threshold'] = pd.Series(10, index=np.arange(1,13)).hvplot.line(c='r', alpha=0.2, line_width=30)
    
    def plot_click_policy(plot, element):
        plot.state.legend.click_policy = 'hide'
    
    f = ndplot.opts(
        legend_position='right', 
        legend_spacing=0,
        toolbar='left', 
        ylabel = 'Specific humidity (g/kg)',
        xlabel = 'Month',
        width = 900,
        height = 450,
        fontsize={'title': 10, 'labels': 10, 'xticks': 9, 'yticks': 9, 'legend': 9},
        title = 'Click on legend entries to hide the corresponding lines',
        hooks=[plot_click_policy])
    
    return f
    
# -------------------------- TOP-LEVEL SLIDER --------------------- #    
    
    
lookback = pn.widgets.DiscreteSlider(options=[
    "L10Y ('12-'21)",
    "L5Y ('17-'21)",
    "L3Y ('19-'21)",
    "L1Y ('21)"
])

def lookback_map(x):
    v = {
        "L10Y ('12-'21)": 'L10Y',
        "L5Y ('17-'21)": 'L5Y',
        "L3Y ('19-'21)": 'L3Y',
        "L1Y ('21)": 'L1Y'
    }[x]
    
    return v
    
    
# -------------------------- WORLD MAP ---------------------------- #

def generate_world_map():

    ds = xr.open_dataset("data/monthly_humidity_lbk.nc")['humidity']#.rio.write_crs(4326)#['humidity']
    ds = np.minimum(ds, 20) # Saturate at 20 g/Kg specific humidity

    custom_hover = bk.models.HoverTool(tooltips=[('Specific Humidity', '@image{0} g/kg')])

    #### EDIT START
    gvds = gv.Dataset(ds)
    images = gvds.to(gv.Image, ['x', 'y'], dynamic=True)

    plot_map = regrid(images).opts(
        cmap='RdBu',
        projection=crs.Robinson(),
        clim=(0,20),
        colorbar=True,
        tools=[custom_hover],
        width=900, 
        height=500) * gv.feature.coastline
    
    return plot_map

plot_map = generate_world_map()

def generate_hawaii_map():
    ds2 = xr.open_dataset("data/hawaii_30y.nc")['humidity'].sel(band = 1)

    custom_hover = bk.models.HoverTool(tooltips=[('Specific Humidity', '@image{0} g/kg')])

    gvds = gv.Dataset(ds2)
    images = gvds.to(gv.Image, ['x', 'y'], dynamic=True)

    base_map = hv.element.tiles.OSM()

    plot_map = (base_map * regrid(images).opts(
        cmap='RdBu',
        clim=(0,20),
        alpha=0.7,
        colorbar=True,
        tools=[custom_hover],
        width=900, 
        height=500)).opts(global_extent=False)
    
    return plot_map

plot_hawaii = generate_hawaii_map()

#@pn.depends(lookback)
#def plot_map(lval):
#    #fig = ds.hvplot(
#    fig = ds.sel(lookback = lookback_map(lval)).hvplot(
#        groupby='month',
#        cmap='RdBu',
#        projection=crs.Robinson(),
#        coastline=True,
#        clim=(0,20),
#        tools=[custom_hover],
#        widget_location="bottom",
#        width=900, 
#        height=500
#    )
#    return fig

#### EDIT END


# -------------------------- US STATES ---------------------------- #

# Load countries
us_states_df = pd.read_csv('data/us_states.csv').rename(columns= {'us_state': 'U.S. state'})

# Define the widget
regions = pn.widgets.Select(options=sorted(us_states_df.region.unique().tolist()), size=selector_elem_size)

@pn.depends(regions, lookback)
def us_state_timeseries(rval, lval):
    tbl = (
        us_states_df.loc[(us_states_df['region'] == rval) & 
                          (us_states_df['lookback'] == lookback_map(lval))]
        .set_index(['month', 'U.S. state'])['humidity']
        .unstack(level=-1)
    )
    return build_chart(tbl)

# -------------------------- US CITIES ---------------------------- #

# Load countries
us_cities_df = pd.read_csv('data/us_cities.csv')

# Define the widget
us_states = pn.widgets.Select(options=sorted(us_cities_df.us_state.unique().tolist()), size=selector_elem_size)

@pn.depends(us_states, lookback)
def us_city_timeseries(uval, lval):
    tbl = (us_cities_df.loc[(us_cities_df['us_state'] == uval) & 
                           (us_cities_df['lookback'] == lookback_map(lval))]
           .set_index(['month', 'city'])['humidity'].unstack(level=-1))   
    return build_chart(tbl)

# -------------------------- COUNTRY SERIES ---------------------------- #

countries_df = pd.read_csv('data/countries.csv')

# Define the widget
continents = pn.widgets.Select(options=sorted(countries_df.continent.unique().tolist()), size=selector_elem_size)

@pn.depends(continents, lookback)
def country_timeseries(cval, lval):
    tbl = (countries_df.loc[(countries_df['continent'] == cval) & 
                            (countries_df['lookback'] == lookback_map(lval))]
           .set_index(['month', 'country'])['humidity'].unstack(level=-1))   
    return build_chart(tbl)

# -------------------------- WORLD CITIES SERIES ---------------------------- #

# Load countries
cities_df = pd.read_csv('data/cities.csv')

# Define the widget
countries = pn.widgets.Select(options=sorted(cities_df.country.unique().tolist()), size=selector_elem_size)

@pn.depends(countries, lookback)
def city_timeseries(cval, lval):
    tbl = (cities_df.loc[(cities_df['country'] == cval) & 
                        (cities_df['lookback'] == lookback_map(lval))]
           .set_index(['month', 'city'])['humidity'].unstack(level=-1))
    return build_chart(tbl)


# -------------------------- LAYOUT ---------------------------- #

# Example: https://github.com/holoviz-demos/bird_migration/blob/master/app.py 

dashboard = pn.Row(
    pn.Row(
        pn.Column(
            pn.Row('# Travel Dashboard'),
            pn.Row("All numbers are expressed in **Specific Humidity**, in grams of Water Vapor per kilogram of air", sizing_mode='stretch_width'),
            pn.Row("*Change the slider on the map to choose a month of the year (1=Jan,...12=Dec) or a lookback*"),
            pn.Row("Specific humidity **10 g/kg** is chosen as the **safety threshold** to stay under"),
            
            background='#F1EEEA'
        ),
        pn.Column(
            pn.Row(pn.layout.Tabs(('World Map', plot_map), ('Hawaii', plot_hawaii))),
            pn.Spacer(height=20),
            pn.Row("*Change the lookback period for the line graphs over the last 10, 5, 3 or 1 year(s)*"),
            pn.Row(lookback),
            pn.Row(pn.layout.Tabs(
                ('U.S. States',
                pn.Row(
                    pn.Row('U.S. Region: ', regions),
                    pn.Row(us_state_timeseries)
                )),
                ('U.S. Cities',
                pn.Row(
                    pn.Row('U.S. State: ', us_states),
                    pn.Row(us_city_timeseries)
                )),
                ('Countries',
                pn.Row(
                    pn.Row('Region: ', continents),
                    pn.Row(country_timeseries)
                )),
                ('World Cities',
                pn.Row(
                    pn.Row('Country: ', countries),
                    pn.Row(city_timeseries)
                ))
            ))
        )
    )
)
dashboard.servable()