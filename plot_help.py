import os

import geopandas as gpd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def add_patch_to_legend(ax, facecolor, edgecolor="None", linestyle='-', handles = None,
                        alpha=1, label='label',leg_title = None, ncol = 1,
                        leg_loc = 'upper right', bbox_to_anchor = (1,1), **kwargs):
    '''
    https://github.com/geopandas/geopandas/issues/660
    # Error in normal writing geopandas : No handles with labels found to put in legend.
    # Alternative solution for "Polygon handles not passing to legend"
    if creating new legend, handles should be none
    if adding multiple patches, handles should be passed

    this should be run just before saving figure
    Args:
        facecolor:
        edgecolor:
        linestyle:
        alpha:
        label:

    Returns:
    handles
    '''
    import matplotlib.patches as mpatches

    pmark = mpatches.Patch(facecolor=facecolor,
                           edgecolor=edgecolor,
                           linestyle=linestyle,
                           alpha=alpha,
                           label=label,
                           **kwargs
                          )
    if handles is None:
        handles, _ = ax.get_legend_handles_labels()

    handles = [*handles,pmark]
    ax.legend(
        handles=handles,
        title=leg_title,
        loc=leg_loc,
        ncol=ncol,
        shadow=True,
        bbox_to_anchor = bbox_to_anchor)
    return handles


def add_marker_to_legend(ax, handles=None, linewidth=None, linestyle=None,color=None,  marker=None,
                         markersize=None,  markeredgewidth=None,
                         markeredgecolor=None, markerfacecolor=None,
                         label='markerlabel', leg_title=None, ncol=1, leg_loc='upper right',
                         bbox_to_anchor=(1, 1), **kwargs):
    '''

    Args:
        ax:
        handles:
        linewidth:
        linestyle:
        color:
        marker:
        markersize:
        markeredgewidth:
        markeredgecolor:
        markerfacecolor:
        alpha:
        label:
        leg_title:
        ncol:
        leg_loc:
        bbox_to_anchor:

    Returns:
    handles
    '''
    from matplotlib.lines import Line2D

    marker = Line2D([0], [0],
                    label=label,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    markeredgewidth=markeredgewidth,
                    markeredgecolor=markeredgecolor,
                    markerfacecolor=markerfacecolor,
                    **kwargs)

    if handles is None:
        handles, _ = ax.get_legend_handles_labels()

    handles = [*handles, marker]
    ax.legend(
        handles=handles,
        title=leg_title,
        loc=leg_loc,
        ncol=ncol,
        shadow=True,
        bbox_to_anchor=bbox_to_anchor)

    return handles

def remove_ctx_annotations(ax):
    '''
    remove annotation from a plot
    when using contextily it will add attribution info to the bottom of the plot
    '''
    annotations = [child for child in ax.get_children() if isinstance(child, mpl.text.Text)]
    annotations[0].remove()


def yearly_dates(ax, major=5, minor=1):
    import matplotlib.dates as mdates

    years = mdates.YearLocator(major)  # every year
    years2 = mdates.YearLocator(minor)  # every year
    # months = mdates.MonthLocator(6)  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(years2)
    # return ax


def baseline(ax, yearstart=None, yearend=None, hard=False):
    '''
    set x-axis date limits to yearstart and yearend
    Args:
        ax:
        yearstart: default is 1980
        yearend: default is current year +1
        hard:

    Returns:

    '''
    import datetime
    import matplotlib.dates as mdb

    if yearend is None:
        yearend = (datetime.datetime.now()).year+1

    if yearstart is None:
        yearstart = 1980

    lims = ax.axis()

    dstart = mdb.date2num(datetime.datetime(yearstart, 1, 1))
    dend = mdb.date2num(datetime.datetime(yearend, 1, 1))

    if (lims[0] < dstart) and not hard:
        dstart = lims[0]

    ax.set_xlim([dstart, dend])


def relabel_bar(ax, freq=5):
    '''
    fix labels on x-axis to have label only every 5 (years)
    '''

    def f(val):
        if (int(float(val.get_text())) % freq) == 0:
            return val.get_text()
        else:
            return ''

    labels = [f(item) for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)


def yrange_(ax, min_range = 150):
    '''
    check if y range is at least min_range. if not then push y-range to min_range/2 in each direction from current lim
    Args:
        ax:
        min_range:

    Returns:

    '''
    lims = ax.axis()
    ytop = lims[3]
    ybot = lims[2]
    if (ytop - ybot) < min_range:
        mid = (ytop + ybot) / 2.
        rang = min_range/2
        yo = [mid - rang, mid + rang]
        ax.set_ylim(yo)
    ax.set_xlabel('')


def add_line(ax, value, start='1/1/1900', end_dt='1/1/2020', periods=2):
    import pandas as pd
    # add mcl line
    xlim = ax.get_xlim()
    ax.plot(pd.date_range(start=start, end=end_dt, periods=2),
            [value, value])
    ax.set_xlim(xlim)


def ylim_conting(ax, ylim):
    '''
	set a new ylimit. only set if greater than current ylimit
	'''
    lims = ax.axis()
    ytop = lims[3]
    if ytop < ylim:
        yo = [0, ylim]
        ax.set_ylim(yo)


def yticklabel_format_comma(ax):
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))


def color_list(unq_list, cmapper=None, spec_colors = None, series = None):
    '''
    make categorical colors for plotting. use as color keyword in matplotlib plots. then insert the legpatch for custom legend.

    function to make dictionary of colors for categorical list

    create color dictionary for list of colors from unq_list
    unq_list is a unique list for whcih colors will be made
    
    the cmapper is a color map from the matplotlib.cm

    spec_colors can be a dict of {key: color}, like {'Vineyard: 'purple'}

    if series is supplied, it will return the colors used for plotting (RGBA)
    
    example:
    
    unq = ['cheese','bread','jam']
    colordict, legpatch = color_list(unq)
    fig,ax = plt.subplots()
    ax.legend(handles = legpatch)
    
    '''
    import matplotlib.patches as mpatches

    if cmapper is None:
        import matplotlib
        cmapper = matplotlib.cm.tab20b

    def get_colors(n):
        return [(name, cmapper(float(cnt) / len(n))) for cnt, name in enumerate(n)]

    colordict = dict(get_colors(unq_list))

    if spec_colors is not None:
        import matplotlib.colors as mcolors
        for key in spec_colors:
            colordict[key] = mcolors.to_rgba(spec_colors[key])

    legpatch = [mpatches.Patch(color=colordict[key], label=key) for key in colordict.keys()]

    if series is None:
        return colordict, legpatch
    else:
        if series.isin(colordict.keys()).all():
            colors = series.apply(lambda x: colordict[x])
            return colordict, legpatch, colors
        else:
            missing = series[~series.isin(colordict.keys())].unique()
            raise AssertionError(f'Value not in colordict. missing\n{missing}\ncolordict:\n{colordict.keys()}')


def get_discrete_colors(ax, bounds=[-10, 0, 10, 210], cmap=mpl.cm.jet, plot=True):
    # https://matplotlib.org/3.1.1/tutorials/colors/colorbar_only.html#discrete-intervals-colorbar

    cmap.set_over('0.25')
    cmap.set_under('0.75')
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    boundaries=[0] + bounds + 10,
                                    extend='both',
                                    ticks=bounds,
                                    spacing='proportional',
                                    orientation='horizontal')
    if plot:
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        cb2.set_label('Discrete intervals, some other units')
        fig.show()

    return cmap, norm


def clip2axis(ax, gdf):
    '''
    clip features to extent of axis. assumes axis is crs=2226
    Args:
        ax:
        gdf:

    Returns:

    '''
    ext = ax.get_extent()
    # ext = ax.get_extent()

    clipped = gpd.clip(gdf.to_crs(2226), [ext[0], ext[2], ext[1], ext[3]])

    # # function to return polygon
    # def bbox(long0, lat0, lat1, long1):
    #     return Polygon([[long0, lat0],
    #                     [long1,lat0],
    #                     [long1,lat1],
    #                     [long0, lat1]])
    #
    # b = bbox(ext[0], ext[2], ext[1], ext[3])
    #
    # clipped = gpd.clip(gdf.to_crs(2226), mask= b )

    return clipped

def label_points(ax,gdf, colname,basin_name=None,iter_limit = 20, fmt =".0f", text_color = 'r', fontsize= 10,
    already_str = True, masked_area = None,
            zorder = 100,   kwds_dict = {}, crs = 2226, arrow_props = None, add_box = False):
    '''

    Args:
        ax:
        gdf:
        colname:
        basin_name: to filter labeling to basins
        fmt: str format
        kwds_dict:
        arrow_props: default dict(arrowstyle='-', color='g')


    Returns:

    '''
    from adjustText import adjust_text
    from matplotlib.transforms import Bbox
    import geopandas as gpd
    from . import rich_gis

    if arrow_props is None:
        arrow_props = dict(arrowstyle='-', color='g')


    gdf = gdf.to_crs(crs)
    gdf = gdf.reset_index(drop=True)
    gdf = clip2axis(ax, gdf)

    if basin_name is None:
        pass
    else:
        print(f'filtering label locations to within {basin_name}')
        bas = rich_gis.get_active_subbasins()
        bas.index = ['PET','SRP', "SON"]
        gdf = gpd.overlay(gdf, bas.loc[basin_name,:])


    if already_str:
        gdf.loc[:,'new_label'] = gdf.loc[:, colname].values
        pass
    else:
        gdf.loc[:,'new_label'] = gdf.loc[:, colname].apply(lambda x: f"{x:{fmt}}").values

    if add_box:
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    else:
        props = None

    bbox_list = []
    if masked_area is None:
        pass
    else:
        geometries = masked_area.to_crs(2226).geometry
        # Create an empty list to store the bbox objects
        bbox_list = []

        # Iterate over the geometries and create bbox objects
        for geom in geometries:
            xmin, ymin, xmax, ymax = geom.bounds
            bbox = Bbox([[xmin, ymin], [xmax, ymax]])
            bbox_list.append(bbox)

    texts = [   ax.text(x.geometry.x, x.geometry.y, x['new_label'],
                color=text_color, fontsize=fontsize,  bbox=props, zorder = zorder)

                for _, x in gdf.iterrows()]

    # fig, ax = plt.subplots()
    # ax.add_collection(paths)

    adjust_text(texts, ax = ax, lim = iter_limit, arrowprops=arrow_props,
                add_objects = bbox_list,
                **kwds_dict)

    return texts

def label_poly(gdf, ax, column="APN", marker='o', color='w', text_color='r', label_exterior = False, fmt = 's'):
    '''
    Label polygon centroids
    Args:
        fmt:
        label_exterior:
        gdf:
        ax:
        column:
        marker:
        color:
        text_color:

    Returns:

    '''
    import shapely

    gdf = clip2axis(ax, gdf)

    _r = gdf.copy().explode() #convert multipolygons to polygons
    if label_exterior:
        # Extract the point values that define the perimeter of the polygon
        def get_ext(geo):
            x, y = geo.exterior.coords.xy
            i = int(len(x) / 2)
            return shapely.geometry.Point(x[i], y[i])

        _r.geometry = _r.geometry.apply(get_ext)

    else:
        _r.geometry = _r.geometry.representative_point()

    label_points(ax, _r, column, fmt=fmt, basin_name=None, text_color=text_color,
                    kwds_dict=dict(arrowprops=dict(arrowstyle='->', color='k')))
    _r.plot(ax=ax, marker=marker, color=color)


def stackedbar_wdates(df, ax=None, fig=None, plot_wet_bars=True, remove_wet_dry=False, ncol=1, return_leg_handles=False,
                      colormap=None, bar_width=365, leg_title=None, historical=True, keep_column_order = False):
    '''
    create a stacked bar plot with dates as axis

    df = pd.DataFrame(index = pd.date_range('1/1/1975',freq='1A', end = '1/1/2000'),data  =np.random.random((25,4))*1e3,columns = ['one','two','three','four'])
    df['one']= np.sin(-df.index.year/1)*1000-2000
    df['two']= np.sin(df.index.year/1)*1000
    fig, ax =plot_help.stackedbar_wdates(df,remove_wet_dry=False)

    remove_wet_dry: create separate wet/dry legend (True: no legend)
    colormap = mpl.cm.nipy_spectral

    '''
    # reorder here so that bars are sorted to be bigger closer to zero.
    neg = df.loc[:, df.sum() > 0].sum().sort_values(ascending=False).index
    pos = df.loc[:, df.sum() < 0].sum().sort_values(ascending=True).index
    ind = neg.append(pos)
    if keep_column_order:
        ind = df.columns.tolist()
        pass
    else:
        df = df.reindex(columns=ind)

    # do similar to above but for legend... a different order here
    neg = df.loc[:, df.sum() > 0].sum().sort_values(ascending=True).index
    pos = df.loc[:, df.sum() < 0].sum().sort_values(ascending=True).index

    if keep_column_order:
        ind_forleg = ind
    else:
        ind_forleg = neg.append(pos)

    df = df.T

    # Take negative and positive data apart and cumulate
    def get_cumulated_array(data, **kwargs):
        cum = data.clip(**kwargs)
        cum = np.cumsum(cum, axis=0)
        d = np.zeros(np.shape(data))
        d[1:] = cum[:-1]
        return d

    data = df.values
    cumulated_data = get_cumulated_array(data, min=0)
    cumulated_data_neg = get_cumulated_array(data, max=0)

    # Re-merge negative and positive data.
    row_mask = (data < 0)
    cumulated_data[row_mask] = cumulated_data_neg[row_mask]
    data_stack = cumulated_data

    if ax == None:
        fig, ax = plt.subplots(figsize=(6, 4))

    def getColor(c, N, idx):  # because default colormap may not be enough, create a new colormap here if necessary
        import matplotlib as mpl
        cmap = mpl.cm.get_cmap(c)
        norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
        return cmap(norm(idx))

    if colormap is None:
        # colormap=mpl.cm.nipy_spectral
        colormap = mpl.cm.tab10

    color_list = [getColor(colormap, df.shape[0], x) for x in range(0, df.shape[0])]
    data_shape = np.shape(data)

    # move columns so that their dates are in center because plotting them from center below
    df.columns = df.columns + df.columns.to_series().diff().bfill() / 2

    for i in np.arange(0, data_shape[0]):
        ax.bar(df.columns, data[i], bottom=data_stack[i], color=color_list[i],
               label=df.index[i], width=bar_width, edgecolor='k', linewidth=.5,
               align='center')

    ax.legend()

    l = ax.get_xlim()
    if plot_wet_bars:
        if historical:
            x = pw.plot_wet_per('SRP', '1-1-1900', '1-1-2020', 1.10, 3, ax, plot_obs=False);
            x.calc_wet()
            x.plot_bars(kind='None', bar_height=10000000, plot_dry=True, calwy_plot=True)
            ax.set_xlim(l)
            ax.legend(bbox_to_anchor=(1, 1), ncol=ncol)
        else:
            print('using future plot')
            from .utils.plot_wet_future import plot_wet_fut
            fig, ax = plot_wet_fut(fig = fig, ax = ax)


    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 1), ncol=ncol)

    new_hands, ind_forleg, hand_wetdry, lab_wetdry = remove_wet_dry_legend(ax,
               legend_order=ind_forleg, remove_wet_dry=remove_wet_dry, leg_title=leg_title, ncol=ncol)

    years = mdates.YearLocator(5)  # every year
    months = mdates.YearLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    ax.set_ylabel('acre-feet')
    ax.set_xlabel('Water Year')

    yticklabel_format_comma(ax)
    # ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    # a# Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')

    if return_leg_handles:
        return fig, ax, new_hands, ind_forleg
    else:
        return fig, ax


def remove_wet_dry_legend(ax,legend_order = None, remove_wet_dry = False, leg_title = '',ncol = 1 ):
    '''
    remove wet/dry from legend, and optionally place wet/dry in separate new legend
    Args:
        ax:
        legend_order: new order for legend
        remove_wet_dry:
        leg_title:
        ncol:

    Returns:

    '''
    # Fix legend
    hand, labl = ax.get_legend_handles_labels()
    lab_remove = ['Wet', 'Very Wet', 'Dry', 'Very Dry']
    handout = []
    lablout = []
    hand_wetdry = []
    lab_wetdry = []
    for h, l in zip(hand, labl):
        if l not in lab_remove:
            lablout.append(l)
            handout.append(h)
        else:
            hand_wetdry.append(h)
            lab_wetdry.append(l)

    if legend_order is None:
        new_hands = handout[::-1] #reverse order here
        legend_order = lablout[::-1] #reverse order here
    else:
        # reorder legend to reflect sorting from above
        zipped = {l: h for l, h in zip(lablout, handout)}
        new_hands = [zipped[n] for n in legend_order]

    # if not removing wet/dry legend, add it first, keep artist, plot artist, then add legend for bars
    if not remove_wet_dry:
        # draw legend temporarily
        first_legend = ax.legend(new_hands, legend_order, bbox_to_anchor=(1, 1.0,), loc='upper left',
                                 ncol=ncol, title=leg_title)
        plt.draw()

        # todo fix finding legend location on plot help.
        # get bottom left location of wet/dry legend in order to plot bars legend below it
        # bbleg2 = first_legend.legendPatch.get_bbox().inverse_transformed(ax.transAxes)
        # bbleg2 = first_legend.legendPatch.get_bbox().transformed(ax.transAxes)
        bbleg2 = first_legend.legendPatch.get_bbox()
        lower_left = bbleg2.corners()[0]


        # temporarily draw wet-dry to get artis
        leg = ax.legend(hand_wetdry, lab_wetdry, title='Water Year Type',
                        ncol=1, bbox_to_anchor=(1.0,0),
                        loc='lower left')
        # finally dray wet/dry with new location
        ax.add_artist(leg)

        # finally draw first legend
        first_legend = ax.legend(new_hands, legend_order, bbox_to_anchor=(1, 1.0,), loc='upper left', ncol=ncol,
                                 title=leg_title)


    else:  # only add legend for bars if remove_wet_dry is True
        ax.legend(new_hands, legend_order, bbox_to_anchor=(1, 1), ncol=ncol, loc='upper left', title=leg_title)

        # ax.legend(handout[::-1], lablout[::-1],bbox_to_anchor = (1,1),ncol = ncol)

    return new_hands, legend_order, hand_wetdry, lab_wetdry

def color_scheme(hand, lab, all_colors = None):
    '''
    use handles and labels to get color scheme form existing plot
    Args:
        hand:
        lab:
        all_colors:

    Returns:
    colors
    '''
    import matplotlib
    colors = {}

    for x in zip(hand, lab):
        if isinstance(x[0],matplotlib.lines.Line2D):
            colors[x[1]] = x[0].get_markeredgecolor()
        else:
            try:
                colors[x[1]] = x[0].patches[1].get_facecolor()
            except:
                colors[x[1]] = x[0].get_facecolor()

    if all_colors is None:
        pass
    else:
        colors = {**all_colors, **colors}

    return colors

def plot_2_dfs_sep(df, df2, maintitle, historical,
                figname,
                out_fold,
                savefig = True,
                title1='Projected (Baseline)',
                title2='Group 1 Scenario',
                fig=None,
                ax=None,
                ax1=None):
    '''
    plot 2 side-by-side plots with stacked bar w dates
    Args:
        df:
        df2:
        maintitle: suptitle
        historical: for background
        figname: output filename
        out_fold: folder
        savefig: True/False
        title1: left title
        title2: right title

    Returns: fig, [ax, ax1], hand
    '''

    if df.columns.equals(df2.columns):
        pass
    else:
        import warnings
        warnings.warn(f'columns of inputs are not the same.\n{df.columns}\n\nand\n{df2.columns}')

    if any([xi is None for xi in [ax, ax1]]):
        fig, (ax, ax1) = plt.subplots(1, 2, sharey=True)
    else:
        pass

    fig, ax, hand, lab1 = stackedbar_wdates(df,
                           return_leg_handles=True,
                           ax=ax,
                           remove_wet_dry=True,
                           historical=historical)

    ax.legend(handles=hand, loc='upper left', bbox_to_anchor=(-.1, -0.2),
              fancybox=False, shadow=False, ncol=2)

    # check if ordering of column sums is the same. if not then reorder by lab1 series and
    if (df.sum().sort_values().index == df2.sum().sort_values().index).all():
        pass
    else:
        print('reordering df2 as it is different than df1')
        df2 = df2.reindex(columns=pd.Series(lab1[::-1]))
        df2.loc[pd.to_datetime('1900/1/1')] = np.linspace(1e12, 1e9, df2.shape[1])
        df2 = df2.sort_index()

    fig, ax1, hand, lab2 = stackedbar_wdates(df2,
                            return_leg_handles=True,
                            ax=ax1,
                            remove_wet_dry=True,
                            historical=historical)

    # check if ordering of column sums is the same.
    if (lab1 == lab2).all():
        pass
    else:
        raise AssertionError(f'order of plotting not the same.\n{lab1}\n{lab2}\n')

    ax1.legend().remove()
    ax1.set_ylabel('')
    yearly_dates(ax1, major=10)
    yearly_dates(ax, major=10)
    baseline(ax, 2020, 2071, True)
    baseline(ax1, 2020, 2071, True)
    ax.set_title(title1)
    ax1.set_title(title2)
    plt.suptitle(f'{maintitle}', fontweight='bold')

    if savefig:
        # save the figure
        plt.savefig(os.path.join(out_fold, figname+'.png'),
                    dpi=300, figsize=(6, 4), bbox_inches='tight')

    return fig, [ax, ax1], hand


def annotate_outside(ax, value, string, box=False):
    '''
    add an arrow outside of axis.
    Args:
        ax:
        value: y location in data values of arrow
        string: string

    Returns: None

    '''
    if box == True:
        bbox_args = dict(boxstyle="round", fc="0.8")
    else:
        bbox_args = None

    arrowprops = dict(arrowstyle="->",
                      connectionstyle="angle,angleA=90,angleB=0,rad=10")

    text = ax.annotate(string,
                       xy=(1, value), xycoords=('axes fraction', "data"),
                       xytext=(2, -.3), textcoords=('offset points', 'axes fraction'),
                       ha="left", va="center",
                       arrowprops=arrowprops,
                       bbox=bbox_args)

    return text

def match_ylim(ax, ax1):
    '''
    match and set the y-lims for two axes
    '''
    ay = ax.get_ylim()
    ay1 = ax1.get_ylim()
    lims = [np.min([ay[0], ay1[0]]), 1.1*np.max([ay[1], ay1[1]])]
    ax.set_ylim(lims)
    ax1.set_ylim(lims)

def plot_model_array(ml,array, title = '', colorbar = True):
    '''
    plot an array on the model instance location of the array
    Args:
        ml:
        array:
        title:
        colorbar:

    Returns:
    fig, ax
    '''

    import flopy
    import cartopy.crs as ccrs
    import contextily as ctx
    from . import rich_gis
    bas = rich_gis.get_active_subbasins()

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.epsg(ml.modelgrid.epsg)},
                            figsize=(11, 8.5))

    mapt = flopy.plot.map.PlotMapView(model=ml, ax=ax)
    mapt.plot_array(array, masked_values=[0], cmap='jet')
    bas.to_crs(ml.modelgrid.epsg).plot(ax=ax, edgecolor='r', lw=1, facecolor='None')
    ctx.add_basemap(ax, crs=ml.modelgrid.epsg, source=ctx.providers.OpenStreetMap.Mapnik)

    if colorbar:
        fig.colorbar((ax.collections[0]))

    ax.set_title(title)

    return fig, ax