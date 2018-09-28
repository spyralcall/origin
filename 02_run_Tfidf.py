from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os


# ファイル名で渡すので、inputにfilename指定して初期化
# max_dfは0.5（半分以上の文書に出現する言葉はいらん）を設定
count_vectorizer = CountVectorizer(input='filename', max_df=0.5, min_df=1, max_features=3000)

# 全ファイルパスを入れた変数でfit_transform
files = ['result_kujou_sud/' + path for path in os.listdir('result_kujou_sud')]
tf = count_vectorizer.fit_transform(files)

# shapeは8(doc count) * 3000(max_features)になっていた
tf.shape
  #=> (8, 3000)

# featue_name一覧を取得
features = count_vectorizer.get_feature_names()
features[0:5]
  #=> ['0213', '10月25日', '13', '1980', '1989']

# index:100〜104までの5つの単語について、各ドキュメントの出現数を出す。
# 8つの文書を読ませたので、8文書*5文字のmatrixになっている
tf.toarray()[:, 100:105]
  #=> array([[0, 0, 0, 0, 0],
  #=>        [0, 0, 0, 0, 0],
  #=>        [0, 0, 0, 1, 0],
  #=>        [0, 1, 1, 0, 0],
  #=>        [0, 0, 0, 0, 0],
  #=>        [0, 0, 0, 0, 0],
  #=>        [0, 1, 0, 3, 1],
  #=>        [1, 0, 0, 1, 0]], dtype=int64)

tf.toarray()[:, 100]
  #=> array([  0,   0,   0,   0,   0,   0, 190,   0], dtype=int64)

features[100]
  #=> 'ジョバンニ'

tf.toarray()[:, features.index('電話')]
  #=> array([  0,   0,   0,   0,   0,   0, 101,   0], dtype=int64)





# normalizeはl2で、sublinear_tfも使う設定で実行してみる
tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)

# fit_transformにCountVectorizerで生成したmatrixを渡せばtfidfが出せる。
tfidf = tfidf_transformer.fit_transform(tf)

# shapeはtfと変わらず
tfidf.shape
  #=> (8, 3000)

# 最大値は0.269...になっている
print(tfidf.toarray().max())
  #=> 0.2694441892392106

# このfeatureは何かというと、正解はオツベルでした
features[111]
  #=> 'オツベル'

# 「風の音」（2ドキュメントに出ている）はこんなくらいのスコア
tfidf.toarray()[:, features.index('保険')]
  #=> array([ 0. ,  0.02869741 , 0. , 0. , 0. , 0. , 0.01849087, 0.])

# 「公開」（4ドキュメントに出ている）はこのくらいのスコア
tfidf.toarray()[:, features.index('電話')]
  #=> array([ 0.03606189, 0. , 0. , 0. , 0. , 0.04528741, 0.01398999, 0.02719194])

