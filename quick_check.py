#!/usr/bin/env python3
import sys,os,pandas as pd,numpy as np;exec('''print("🔍 QUICK TRAINING CHECK");print("="*40);files=["self_supervised_model.py","utils.py","../data/training_prepared_data.csv"];all_ok=True;[print(f"✅ {f}") if os.path.exists(f) else [print(f"❌ {f}"),setattr(sys.modules[__name__],"all_ok",False)] for f in files];print("\\n📦 Testing imports...");[exec(f"import {m};print(f'✅ {m}')") for m in ["tensorflow","keras","utils","sklearn"] if exec(f"try:\n import {m}\nexcept Exception as e:\n print(f'❌ {m}: {{e}}')\n all_ok=False",globals())];print("\\n📊 Checking data...");exec('''try:
df=pd.read_csv("../data/training_prepared_data.csv")
print(f"✅ Data: {len(df)} samples")
if 'image_path' not in df.columns:print("❌ Missing image_path column");all_ok=False
else:print("✅ Required columns OK")
except Exception as e:print(f"❌ Data error: {e}");all_ok=False''');print("\\n🖥️ Hardware check...");exec('''try:
import tensorflow as tf
gpus=tf.config.experimental.list_physical_devices('GPU')
print(f"✅ Found {len(gpus)} GPU(s)" if gpus else "ℹ️ CPU mode")
except:print("❌ Hardware check failed")''');print("\\n"+"="*40);print("🎉 READY TO TRAIN!" if all_ok else "❌ Fix issues first");print("Run: python self_supervised_model.py" if all_ok else "Check errors above");print("="*40)''')
