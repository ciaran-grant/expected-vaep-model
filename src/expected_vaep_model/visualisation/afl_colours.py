import matplotlib.pyplot as plt

team_colours = {
    'Adelaide':{
        'primary':'#002b5c',
        'secondary':'#bb9e41'},
    'Brisbane':{
        'primary':'#7C003E',
        'secondary':'#0054A4'},
    'Carlton':{
        'primary':'#1d5a8b',
        'secondary':'#bbbfc0'},
    'Collingwood':{
        'primary':'white',
        'secondary':'#494949'},
    'Essendon':{
        'primary':'#CC2031',
        'secondary':'#939598'},
    'Fremantle':{
        'primary':'#664985',
        'secondary':'#a2acb4'},
    'Geelong':{
        'primary':'#1c3c63',
        'secondary':'white'},
    'Gold Coast':{
        'primary':'#B90A34',
        'secondary':'#d3bb51'},
    'Greater Western Sydney':{
        'primary':'#f47a1a',
        'secondary':'#944712'},
    'Hawthorn':{
        'primary':'#fbbf15',
        'secondary':'#4d2004'},
    'Melbourne':{
        'primary':'#002076',
        'secondary':'#cc2031'},
    'North Melbourne':{
        'primary':'#013b9f',
        'secondary':'white'},
    'Port Adelaide':{
        'primary':'#008aab',
        'secondary':'#3b3b3b'},    
    'Richmond':{
        'primary':'#fed102',
        'secondary':'#4d4d4d'},
    'St Kilda':{
        'primary':'#ed0f05',
        'secondary':'#4d4d4d'},
    'Sydney':{
        'primary':'#ed171f',
        'secondary':'white'},
    'West Coast':{
        'primary':'#062ee2',
        'secondary':'#ffd700'},
    'Western Bulldogs':{
        'primary':'#014896',
        'secondary':'#c70136'}
}


team_colourmaps = {
    'Adelaide':plt.cm.Blues,
    'Brisbane Lions':plt.cm.BuPu,
    'Carlton':plt.cm.Blues,
    'Collingwood':plt.cm.Greys,
    'Essendon':plt.cm.Reds,
    'Fremantle':plt.cm.Purples,
    'Geelong':plt.cm.Blues,
    'Gold Coast':plt.cm.YlOrRd,
    'Greater Western Sydney':plt.cm.Oranges,
    'Hawthorn':plt.cm.YlOrBr,
    'Melbourne':plt.cm.PuBu,
    'North Melbourne':plt.cm.Blues,
    'Port Adelaide':plt.cm.GnBu,    
    'Richmond':plt.cm.YlGn_r,
    'St Kilda':plt.cm.Reds,
    'Sydney':plt.cm.Reds,
    'West Coast':plt.cm.YlGnBu,
    'Western Bulldogs':plt.cm.PuBu
}