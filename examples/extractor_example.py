from src.extractors.extractor_aquis import AquisExtractor

def main():
    extractor = AquisExtractor("ES0113900J37")
    df = extractor.extract()

    print("Primeiras 5 linhas do DataFrame:")
    print(df.head())

if __name__ == "__main__":
    main()