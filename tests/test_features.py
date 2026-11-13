from src.data.db_config import get_engine
from src.features.build_features import (
    get_training_data_from_db,
    FeatureEngineer,
    HaversineDistanceTransformer,
    build_preprocessor
)

engine = get_engine()
df_raw = get_training_data_from_db(engine)

# Apply transformers
dist_trans = HaversineDistanceTransformer('seller_lat', 'seller_lng', 'customer_lat', 'customer_lng')
feat_eng = FeatureEngineer()

df_feat = dist_trans.fit_transform(df_raw)
df_feat = feat_eng.fit_transform(df_feat)

print("Engineered columns:", [c for c in df_feat.columns if c not in df_raw.columns])
print("Distance summary stats:\n", df_feat['seller_customer_distance_km'].describe())