from paper_review.ingestion.pipeline import fetch_paper, extract_pages

pdf_path, title = fetch_paper("1706.03762")
print("title:", title)

pages = extract_pages(pdf_path)
print("page count:", len(pages))
print("--- page 13 raw ---")
print(pages[12][:600])