

import IPython.display as display
from matplotlib import pyplot as plt
import pandas as pd
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import warnings
from flask import Flask, request, Response
from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

anime = pd.read_csv('anime.csv')
rating = pd.read_csv('rating.csv')

rating.info()

rating.head()

rating.hist()

anime.info()

anime.head()

figsize = (12, 1.2 * len(anime['type'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(anime, x='rating', y='type', inner='box', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)

anime.plot(kind='scatter', x='rating', y='members', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

anime['rating'].plot(kind='hist', bins=20, title='rating')
plt.gca().spines[['top', 'right',]].set_visible(False)

types = anime['type'].value_counts()
fig, ax = plt.subplots()
ax.pie(types, labels=types.index, autopct="%1.1f%%")
ax.set_title("Anime Types")

#matplotlib grafiklerini gösterme
# plt.show()

df = anime.copy()
# members sütunundaki verileri ölçeklendir
scaler = StandardScaler()
members_scaled = scaler.fit_transform(df[['members']])

# K-means kümeleme algoritması ile en uygun grup sayısını belirlemek için "dirsek yöntemi"ni kullan
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(members_scaled)
    wcss.append(kmeans.inertia_)

# Dirsek grafiğini çiz
plt.plot(range(1, 11), wcss)
plt.title('Dirsek Yöntemi')
plt.xlabel('Küme Sayısı')
plt.ylabel('WCSS')
# plt.show()



# K-means algoritması ile üyeleri gruplara ayır
#KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')[source]

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['members_group'] = kmeans.fit_predict(members_scaled)

# Grupları inceleyin
print(df['members_group'].value_counts())

# Her kolon için gruplama işlemi uygulayın
#grouped_data = df.groupby('members_group').mean().reset_index()


df[df.rating.isna()] = df.rating.mean()
df[df.rating.isna()]
df['type_x'], _= pd.factorize(df['type'])
df.type_x.unique()

class ContentBase(object):
  def __init__(self, anime):
    self.anime_CB = anime.copy()
    self.sparse_matrix_CB = None
    self.df_zero_shot = anime[['anime_id']].copy()
    self.knn_CB = None
    # En uygun grup sayısını belirle (örneğin, dirsek grafiğinden 3)
    self.optimal__member_clusters = 4
    self.__fit_CB()

  def __preprocess_data_CB(self):
    self.anime_CB.genre.fillna('Unknown', inplace=True)
    self.anime_CB.type.fillna('Unknown', inplace=True)
    self.anime_CB.episodes.replace('Unknown', 0, inplace=True)

  def __genre_to_dummies(self):

    genres = self.anime_CB.genre.str.split(", ", expand=True)
    unique_genres = pd.Series(genres.values.ravel('K')).dropna().unique()
    dummies = pd.get_dummies(genres)
    for genre in unique_genres:
      self.df_zero_shot["Genre: " + genre] = dummies.loc[:, dummies.columns.str.endswith(genre)].sum(axis=1)

  def __type_to_dummies(self):
    type_dummies = pd.get_dummies(self.anime_CB.type, prefix="Type:", prefix_sep=" ")
    self.df_zero_shot = pd.concat([self.df_zero_shot, type_dummies], axis=1)
    self.df_zero_shot['Type: Movie'] = self.df_zero_shot['Type: Movie'].astype('int64')
    self.df_zero_shot['Type: Music'] = self.df_zero_shot['Type: Music'].astype('int64')
    self.df_zero_shot['Type: ONA'] = self.df_zero_shot['Type: ONA'].astype('int64')
    self.df_zero_shot['Type: OVA'] = self.df_zero_shot['Type: OVA'].astype('int64')
    self.df_zero_shot['Type: TV'] = self.df_zero_shot['Type: TV'].astype('int64')
    self.df_zero_shot['Type: Special'] = self.df_zero_shot['Type: Special'].astype('int64')
    self.df_zero_shot['Type: Unknown'] = self.df_zero_shot['Type: Unknown'].astype('int64')

  def __rating_to_dummies(self):
    self.df_zero_shot['rating'] = self.anime_CB['rating']
    self.df_zero_shot[self.df_zero_shot.rating.isna()] = self.df_zero_shot['rating'].mean()

  def __episodes_to_dummies(self):
    self.df_zero_shot['episodes'] = self.anime_CB['episodes']
    self.df_zero_shot.episodes = self.df_zero_shot['episodes'].astype('int64')
    # 'episodes' kolonundaki minimum ve maksimum değerleri bulun
    min_val = self.df_zero_shot.episodes.min()
    max_val = self.df_zero_shot.episodes.max()

    # 'episodes' kolonundaki değerleri normalize etmek
    self.df_zero_shot.episodes = (self.df_zero_shot.episodes - min_val) / (max_val - min_val)

  def __members_to_dummies(self):
    scaler = StandardScaler()
    members_scaled = scaler.fit_transform(self.anime_CB[['members']])
    kmeans = KMeans(n_clusters=self.optimal__member_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    self.df_zero_shot['members_group'] = kmeans.fit_predict(members_scaled)
    members_dummies = pd.get_dummies(self.df_zero_shot.members_group, prefix="Group:", prefix_sep=" ")
    self.df_zero_shot = pd.concat([self.df_zero_shot, members_dummies], axis=1)
    self.df_zero_shot.drop('members_group', axis=1, inplace=True)
    self.df_zero_shot['Group: 0'] = self.df_zero_shot['Group: 0'].astype('int64')
    self.df_zero_shot['Group: 1'] = self.df_zero_shot['Group: 1'].astype('int64')
    self.df_zero_shot['Group: 2'] = self.df_zero_shot['Group: 2'].astype('int64')
    self.df_zero_shot['Group: 3'] = self.df_zero_shot['Group: 3'].astype('int64')



  def __create_sparse_matrix_CB(self):
    self.df_zero_shot.drop(['anime_id'], axis=1, inplace=True)
    self.sparse_matrix_CB = csr_matrix(self.df_zero_shot.values)

  def __create_knn_model_CB(self, n : int = 10):
    self.knn_CB = NearestNeighbors(metric='cosine', algorithm='brute')
    self.knn_CB.fit(self.sparse_matrix_CB)

  def __fit_CB(self):
    self.__preprocess_data_CB()
    self.__genre_to_dummies()
    self.__type_to_dummies()
    self.__rating_to_dummies()
    self.__episodes_to_dummies()
    self.__members_to_dummies()
    self.__create_sparse_matrix_CB()
    self.__create_knn_model_CB()

  def __clean_result_CB(self,indices, score, id):
    score = list(score[0])
    indices = list(indices[0])
    index_of_number = indices.index(id)  # Sayının indeksini bulma
    indices.pop(index_of_number)
    score.pop(index_of_number)
    return indices, score

  def normalize_score_CB(self, result):
    # Z-Score Normalizasyonu
    mean_score = result['score'].mean()
    std_score = result['score'].std()

    result['score'] = (result['score'] - mean_score) / std_score

    return result

  def get_recommendations_CB(self, anime_id : int, n : int = 10):
    index = self.anime_CB[self.anime_CB.anime_id == anime_id].index[0]
    score, indices = self.knn_CB.kneighbors(self.sparse_matrix_CB[index], n_neighbors=n+1)
    indices, score = self.__clean_result_CB(indices, score, index)
    recommend = self.anime_CB.loc[self.anime_CB.index[indices]]
    recommend.set_index('anime_id', inplace=True)
    recommend['score'] = score
    recommend = self.normalize_score_CB(recommend)
    return recommend

class ItemUserCF(object):
  def __init__(self, rating, anime):
    self.rating = rating.copy()
    self.anime = anime.copy()
    self.anime.set_index('anime_id', inplace=True)
    self.sparse_matrix = None
    self.knn = None
    self.anime_indexes = None
    self.__fit()

  def __fit(self):
    self.__preprocess_data()
    self.__create_sparse_matrix()
    self.__create_knn_model()

  def __preprocess_data(self):
    #Replacing -1 with NaN in user_rating column
    self.rating["rating"].replace({-1: np.nan}, inplace=True)
    counts = self.rating['user_id'].value_counts()
    self.rating= self.rating[self.rating['user_id'].isin(counts[counts >= 200].index)]

    #dropping all the null values as it aids nothing
    self.rating = self.rating.dropna(axis = 0, how ='any')


  def __create_sparse_matrix(self):
    pivot = self.rating.pivot_table(index='anime_id',columns='user_id',values='rating').fillna(0)
    self.anime_indexes = pivot.index
    self.sparse_matrix = csr_matrix(pivot.values)

  def __create_knn_model(self, n : int = 10):
    self.knn = NearestNeighbors(metric='euclidean', algorithm='brute')
    self.knn.fit(self.sparse_matrix)

  def __clean_result(self,indices, score, id):
    score = list(score[0])
    indices = list(indices[0])
    index_of_number = indices.index(id)  # Sayının indeksini bulma
    indices.pop(index_of_number)
    score.pop(index_of_number)
    return indices, score

  def normalize_score_CF(self, result):

    # Z-Score Normalizasyonu
    mean_score = result['score'].mean()
    std_score = result['score'].std()

    result['score'] = (result['score'] - mean_score) / std_score

    return result

  def get_recommendations(self, anime_id : int, n : int = 10):
    if anime_id in self.anime_indexes:
      index =self.anime_indexes.get_loc(anime_id)
      score, indices = self.knn.kneighbors(self.sparse_matrix[index], n_neighbors=n+1)
      indices, score = self.__clean_result(indices, score, index)
      recommend = self.anime.loc[self.anime_indexes[indices]]
      recommend['score'] = score
      recommend = self.normalize_score_CF(recommend)
    else:
      recommend = pd.DataFrame(columns=self.anime.columns)
      recommend['score'] = pd.Series(dtype='float64')

    return recommend


class Hybrit(ContentBase, ItemUserCF):
  def __init__(self, rating, anime):
    ContentBase.__init__(self, anime)
    ItemUserCF.__init__(self, rating, anime)


  def get_recommendations_hyrid(self, anime_id : int, n : int = 10, w1 = 0.5, w2= 0.5):
    try:
      content_base = self.get_recommendations_CB(anime_id, n)
      item_user_cf = self.get_recommendations(anime_id, n)
      item_user_cf.score = item_user_cf.score * w1
      content_base.score = content_base.score * w2
      content_base['type_recommend'] = 'content_base'
      item_user_cf['type_recommend'] = 'item_user'
      content_base.reset_index(inplace=True)
      item_user_cf.reset_index(inplace=True)
      #combined_table = pd.concat([content_base, item_user_cf])
      result = pd.merge(content_base, item_user_cf, on='anime_id', suffixes=('_df1', '_df2'), how='outer')
      result['score'] = result[['score_df1', 'score_df2']].mean(axis=1)
      #result = combined_table.sort_values('score', ascending=True)
      #result.drop_duplicates(subset='anime_id', keep='first', inplace=True)
      result.reset_index(drop=True, inplace=True)
      return result
    except:
      print("Invalid anime ID")
  






def search_anime(df, search_term):
    # Case insensitive arama ve eksik harfleri göz ardı etmek için regex kullanımı
    pattern = re.compile(search_term, re.IGNORECASE)

    # DataFrame'de arama
    matching_animes = df[df['name'].str.contains(pattern)]
    return matching_animes

# Kullanıcıdan arama terimi alalım
# search_term = "haikyu"

# Arama sonuçlarını getirelim
# result = search_anime(anime, search_term)

# print("\nSearch Results")
# result

# anime[anime.anime_id == 32935]

# hybrit = Hybrit(rating, anime)

# hybrit.get_recommendations_hyrid(32935, w1=0.55, w2=0.45)



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def top_20_anime():
    file_path = 'anime.csv'
    anime = pd.read_csv(file_path)

    search_query = request.form.get('search_query', '')
    if request.form.get('reset'):
        search_query = ''

    if search_query:
        filtered_anime = anime[anime['name'].str.contains(search_query, case=False, na=False)]
    else:
        filtered_anime = anime.sort_values(by='rating', ascending=False).head(20)

    filtered_anime = filtered_anime[['anime_id', 'name', 'rating']].reset_index(drop=True)

    html_template = '''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>Top 20 Highest Rated Anime</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
        <style>
          .modal-content {
            min-width: fit-content;
          }
          .modal-body {
            max-height: 800px;
            overflow-y: auto;
          }
        </style>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
      $(document).ready(function(){
        $('table tbody tr').click(function(){
          var animeId = parseInt($(this).find('td:eq(0)').text(), 10);

          $.ajax({
            url: '/search',
            type: 'POST',
            data: {search_term: animeId},
            success: function(response){
              var results = response.results;
              console.log(results);
              var modalBody = $('#resultsModal .modal-body');
              modalBody.empty();
              if (results.length > 0) {
                console.log(results); // Log the results to debug
                    var table = '<table class="table table-striped"><thead><tr><th>Anime ID</th><th>Name</th><th>Genre</th><th>Type</th><th>Rating</th><th>Type Recommend</th></tr></thead><tbody>';
                results.forEach(function(anime){
                      table += '<tr><td>' + anime.anime_id + '</td><td>' + anime.name_df1 + '</td><td>' + anime.genre_df1 + '</td><td>' + anime.type_df1 + '</td><td>' + anime.rating_df2 + '</td><td>' + anime.type_recommend_df2 + '</td></tr>';
                });
                table += '</tbody></table>';
                modalBody.append(table);
              } else {
                modalBody.append('<p>No results found</p>');
              }
              $('#resultsModal').modal('show');
            },
            error: function(xhr, status, error) {
              console.error('AJAX Error: ' + status + error);
            }
          });
        });
      });
    </script>
      </head>
      <body>
        <div class="container mt-5">
          <h1 class="mb-4">Top 20 Highest Rated Anime</h1>
          <form method="post" class="mb-4">
            <div class="form-group">
              <label for="search_query">Search Anime by Name:</label>
              <input type="text" class="form-control" id="search_query" name="search_query" placeholder="Enter anime name">
            </div>
            <button type="submit" class="btn btn-primary" name="search">Search</button>
            <button type="submit" class="btn btn-secondary" name="reset">Reset</button>
          </form>
          <table class="table table-striped">
            <thead>
              <tr>
                <th>Anime ID</th>
                <th>Name</th>
                <th>Rating</th>
              </tr>
            </thead>
            <tbody>
              {% for row in filtered_anime.itertuples() %}
              <tr>
                <td>{{ row.anime_id }}</td>
                <td>{{ row.name }}</td>
                <td>{{ row.rating }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>

        <div class="modal fade" id="resultsModal" tabindex="-1" role="dialog" aria-labelledby="resultsModalLabel" aria-hidden="true">
          <div class="modal-dialog" role="document">
            <div class="modal-content">
              <div class="modal-header">
                <h5 class="modal-title" id="resultsModalLabel">Search Results</h5>
                <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                  <span aria-hidden="true">&times;</span>
                </button>
              </div>
              <div class="modal-body">
              </div>
              <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
              </div>
            </div>
          </div>
        </div>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
      </body>
    </html>
    '''
    # Render HTML with filtered_anime DataFrame
    return render_template_string(html_template, filtered_anime=filtered_anime)

@app.route('/search', methods=['POST'])
def search():
    file_path = 'anime.csv'
    anime = pd.read_csv(file_path)

    search_term = request.form.get('search_term', '')

    try:
      parsed_animeId = int(search_term)
    except ValueError:
    # Handle the case where the search term is not a valid integer
      parsed_animeId = 0
      print(f"Invalid anime ID: {search_term}")

    if search_term:
        hybrit = Hybrit(rating, anime)
        
        results = hybrit.get_recommendations_hyrid(parsed_animeId, n = 10, w1=0.55, w2=0.45)
        print("results")
        print(results)

        # results = results[['anime_id', 'name_df1','genre_df1','type_df1','rating_df2','type_recommend_df2']].where(pd.notnull(results), None).to_dict(orient='records') 

        results = results[['anime_id', 'name_df1', 'genre_df1', 'type_df1', 'rating_df2', 'type_recommend_df2']].replace({np.nan: None})
        
        # Convert the DataFrame to a dictionary
        results_dict = results.to_dict(orient='records')

        # Return the results as JSON
        return jsonify(results=results_dict)
        # return jsonify(results=results)
    return jsonify(results=[])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
