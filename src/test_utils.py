def search_cols(df, keywords=None):

    if keywords:
        for keyword in keywords:
            matches = [
                c for c in df
                if keyword.lower() in c.lower()
            ]
        
            print(f"\n{keyword}:")
            print(matches)

    else:
        raise ValueError(f"No keywords passed to function: keywords={keywords}")

    

