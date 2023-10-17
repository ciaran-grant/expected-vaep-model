import matplotlib.pyplot as plt

team_colours = {
    'Adelaide':{
        'positive':'#002b5c',
        'negative':'#bb9e41'},
    'Brisbane Lions':{
        'positive':'#7C003E',
        'negative':'#0054A4'},
    'Carlton':{
        'positive':'#1d5a8b',
        'negative':'#bbbfc0'},
    'Collingwood':{
        'positive':'white',
        'negative':'#494949'},
    'Essendon':{
        'positive':'#CC2031',
        'negative':'#939598'},
    'Fremantle':{
        'positive':'#664985',
        'negative':'#a2acb4'},
    'Geelong':{
        'positive':'#1c3c63',
        'negative':'white'},
    'Gold Coast':{
        'positive':'#B90A34',
        'negative':'#d3bb51'},
    'Greater Western Sydney':{
        'positive':'#f47a1a',
        'negative':'#944712'},
    'Hawthorn':{
        'positive':'#fbbf15',
        'negative':'#4d2004'},
    'Melbourne':{
        'positive':'#002076',
        'negative':'#cc2031'},
    'North Melbourne':{
        'positive':'#013b9f',
        'negative':'white'},
    'Port Adelaide':{
        'positive':'#008aab',
        'negative':'#3b3b3b'},    
    'Richmond':{
        'positive':'#fed102',
        'negative':'#4d4d4d'},
    'St Kilda':{
        'positive':'#ed0f05',
        'negative':'#4d4d4d'},
    'Sydney':{
        'positive':'#ed171f',
        'negative':'white'},
    'West Coast':{
        'positive':'#062ee2',
        'negative':'#ffd700'},
    'Western Bulldogs':{
        'positive':'#014896',
        'negative':'#c70136'}
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