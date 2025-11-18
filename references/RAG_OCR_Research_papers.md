## OCR for pdf parsing in a RAG pipeline for Research Papers with 2-Column Layouts, Tables, and Charts

Based on your specific requirements for parsing research papers with two-column formats, tables, and charts, here are the optimal solutions:

### Recommended: Nougat (Top Choice for Academic Papers)

**Nougat** (Neural Optical Understanding for Academic Documents) is specifically designed for academic papers and is your **best overall choice**.[^2_1][^2_2][^2_3][^2_4]

#### Why Nougat Excels for Research Papers

**Purpose-built for academic documents:** Trained on 8+ million academic papers from arXiv, PubMed Central, and other scholarly sources[^2_2][^2_3]

**Handles 2-column layouts natively:** Unlike Tesseract which struggles with multi-column documents, Nougat was specifically designed to understand academic paper structures including two-column layouts[^2_5][^2_6][^2_7][^2_8][^2_1]

**Preserves complex elements:**

- Mathematical equations → LaTeX format
- Tables → Structured markdown/HTML
- Charts and figures → Referenced with captions
- Formulas and inline math → Properly formatted[^2_9][^2_1][^2_2]

**Output format:** Converts to MultiMarkdown (.mmd), which integrates seamlessly into RAG pipelines and maintains document structure[^2_2][^2_9]

#### Implementation

```python
# Install Nougat
!pip install nougat-ocr

# CLI usage
nougat path/to/research_paper.pdf -o output_directory

# Python API
from nougat import NougatModel
from pdf2image import convert_from_path

model = NougatModel.from_pretrained("facebook/nougat-base")
pdf_images = convert_from_path("paper.pdf")

# Process each page
for page_img in pdf_images:
    markdown_output = model(page_img)
    print(markdown_output)
```

**Batch Processing for Multiple Papers:**

```python
import os

nougat_cmd = "nougat --markdown --out 'output_dir'"
pdf_path = '/papers'

for pdf in os.listdir(pdf_path):
    os.system(f"{nougat_cmd} /papers/{pdf}")
```


#### Performance Characteristics

**Accuracy:** State-of-the-art for academic papers, especially those with mathematical content[^2_10][^2_1]

**Speed:** ~30 seconds for a 6-page document on CPU (i5 laptop), faster on GPU[^2_10]

**Memory:** Can require 7GB+ RAM for complex papers[^2_10]

### Alternative: Marker (Fast \& Accurate)

**Marker** is an excellent alternative, particularly when speed matters.[^2_11][^2_12][^2_13]

#### Why Consider Marker

**Optimized for scientific papers and books:** Explicitly designed for these document types[^2_12][^2_11]

**Faster than Nougat:** 4x faster processing speed while maintaining accuracy[^2_11][^2_10]

**Comprehensive table and image handling:**

- Extracts images and saves separately
- Formats tables with structure preservation
- Optional LLM enhancement for complex tables (0.907 accuracy with `--use_llm` flag)[^2_12]

**Robust 2-column support:** Uses Surya layout detection to correctly identify reading order in multi-column documents[^2_11][^2_12]

**Benchmark Performance on Scientific Papers:**[^2_12]

- Heuristic score: 96.67
- LLM score: 4.35/5
- Superior to LlamaParse, Mathpix, and Docling on academic content


#### Implementation

```python
# Install
pip install marker-pdf

# CLI usage
marker_single path/to/paper.pdf output_folder --batch_multiplier 2

# Batch processing
marker path/to/papers_directory output_folder --workers 4

# With LLM enhancement for complex tables
marker_single paper.pdf output_folder --use_llm
```

**Python Integration:**

```python
from marker.convert import convert_single_pdf
from marker.models import load_all_models

# Load models once
model_lst = load_all_models()

# Convert PDF
full_text, images, out_meta = convert_single_pdf(
    "research_paper.pdf",
    model_lst,
    max_pages=None,
    langs=["English"]
)

# full_text is markdown with preserved structure
# images is a dict of extracted figures
```


### Alternative: LlamaParse (Premium Mode for Complex Layouts)

For production RAG systems processing research papers, **LlamaParse with Premium Mode** is highly effective.[^2_14][^2_15][^2_16][^2_17]

#### Advantages for Academic Papers

**Premium mode features specifically for complex documents:**[^2_16]

- OCR for scanned papers
- Image extraction with captioning
- Table heading identification
- LaTeX output for equations
- Mermaid format for diagrams

**Parsing instructions:** Customize extraction with natural language prompts[^2_16]

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="your_key",
    result_type="markdown",
    premium_mode=True,  # Essential for 2-column layouts
    parsing_instruction="""
    This is an academic research paper with 2-column layout.
    Preserve the reading order across columns.
    Extract all tables with structure maintained.
    Convert mathematical equations to LaTeX.
    Identify and caption all figures and charts.
    """
)

documents = parser.load_data("research_paper.pdf")
```

**Considerations:**

- API-based (cloud service)
- Premium mode: 1 credit per page
- Excellent for production but costs scale with volume[^2_16]


### Alternative: Docling (Self-Hosted \& Open Source)

**Docling** from IBM Research is ideal if you need a self-hosted solution.[^2_18][^2_19][^2_20][^2_21]

#### Strengths for Research Papers

**Layout-aware processing:** Preserves hierarchical structure including headers, sections, and multi-column layouts[^2_20][^2_18]

**Advanced table extraction:** Uses TableFormer model specifically trained for academic table structures[^2_19][^2_20]

**Cross-page table handling:** Reconstructs tables split across pages[^2_20]

**VLM pipeline option:** Can use vision-language models (GraniteDocling) for enhanced understanding[^2_22][^2_23]

#### Implementation

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("research_paper.pdf")

# Export to markdown preserving structure
markdown = result.document.export_to_markdown()

# Extract specific elements
for element in result.document.body.elements:
    if element.label == "table":
        print(f"Found table: {element.text}")
    elif element.label == "section_header":
        print(f"Section: {element.text}")
```

**Benchmarks (H100 GPU):**[^2_12]

- Processing time: 3.7 seconds/page
- Accuracy on scientific papers: 92.14 (heuristic), 3.72/5 (LLM judge)
- Free and open-source (MIT license)


### Alternative: Unstructured.io (Enterprise-Grade)

For complex production environments with diverse document types.[^2_24][^2_25][^2_26]

#### Key Features for Academic Papers

**Hi-Res strategy:** Computer vision-based layout analysis ideal for complex PDFs[^2_27][^2_24]

**Table Transformer model:** Dedicated model for table structure recognition[^2_25][^2_24]

**VLM strategy:** Uses vision-language models for exceptional accuracy on multi-level headers and nested tables[^2_24]

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="research_paper.pdf",
    strategy="hi_res",  # or "auto" for cost optimization
    infer_table_structure=True,
    extract_images_in_pdf=True,
    model_name="yolox"  # Best for table extraction
)

# Filter for tables
tables = [el for el in elements if el.category == "Table"]
for table in tables:
    print(table.metadata.text_as_html)  # Structured HTML output
```


### Comparison for Research Papers

| Tool | 2-Column Handling | Table Quality | Math/Equations | Speed | Cost | Best For |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **Nougat** | Excellent[^2_1][^2_3] | Very Good | Excellent (LaTeX)[^2_2][^2_9] | Moderate | Free | Academic papers with heavy math |
| **Marker** | Excellent[^2_11][^2_12] | Excellent[^2_12] | Very Good | Fast (4x Nougat)[^2_11][^2_10] | Free | Large-scale academic corpus |
| **LlamaParse** | Excellent[^2_14][^2_15] | Excellent | Very Good | Slow | \$\$ | Production RAG with mixed docs |
| **Docling** | Very Good[^2_18][^2_19] | Very Good[^2_19] | Good | Moderate | Free | Self-hosted requirements |
| **Unstructured** | Very Good[^2_24] | Excellent[^2_24][^2_25] | Good | Fast | Free/Paid tiers | Enterprise pipelines |

### Practical Recommendation

**For your specific use case (research papers with 2-column layouts, tables, and charts):**

1. **Start with Nougat** if your papers have significant mathematical content or you need the highest accuracy on academic formatting[^2_3][^2_1][^2_2]
2. **Use Marker** if you're processing many papers and need speed, or if tables are more critical than complex equations[^2_11][^2_10][^2_12]
3. **Choose LlamaParse Premium** if you're building a production RAG system and can afford API costs for guaranteed quality[^2_15][^2_14][^2_16]
4. **Deploy Docling** if data privacy requires self-hosting or you need deep customization of the parsing pipeline[^2_18][^2_19][^2_20]

### Handling 2-Column Layout Challenges

All recommended tools handle 2-column layouts, but here are specific considerations:

**Tesseract requires manual configuration:**[^2_8]

```python
# Set page segmentation mode to handle columns
pytesseract.image_to_string(image, config='--psm 1')
# psm 1 = Automatic page segmentation with OSD (includes column detection)
```

However, Tesseract still struggles compared to the specialized tools above.[^2_7][^2_5]

**Why specialized tools work better:** They use layout analysis models trained on academic papers to understand:

- Reading order across columns
- Column boundaries
- Figure/table placement that interrupts columns
- Headers/footers spanning both columns[^2_28][^2_12]


### Integration into RAG Pipeline

After parsing with your chosen tool, follow this pattern:

```python
# 1. Parse with Nougat/Marker/etc.
parsed_markdown = parse_research_paper("paper.pdf")

# 2. Semantic chunking for academic content
from langchain.text_splitter import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Title"),
        ("##", "Section"),
        ("###", "Subsection")
    ]
)
chunks = splitter.split_text(parsed_markdown)

# 3. Preserve academic structure in metadata
for chunk in chunks:
    chunk.metadata["document_type"] = "research_paper"
    chunk.metadata["has_tables"] = check_for_tables(chunk.text)
    chunk.metadata["has_equations"] = check_for_latex(chunk.text)

# 4. Embed and store
vectorstore.add_documents(chunks)
```

This approach leverages the structural understanding from OCR tools while maintaining academic document semantics throughout your RAG pipeline.
<span style="display:none">[^2_29][^2_30][^2_31][^2_32][^2_33][^2_34][^2_35][^2_36][^2_37][^2_38][^2_39][^2_40][^2_41]</span>

<div align="center">⁂</div>

[^2_1]: https://parsio.io/blog/extracting-data-from-pdfs-using-ai-claude-3-donut-and-nougat/

[^2_2]: https://www.e2enetworks.com/blog/nougat-neural-optical-understanding-for-academic-documents

[^2_3]: https://openreview.net/forum?id=fUtxNAKpdV

[^2_4]: https://arxiv.org/abs/2308.13418

[^2_5]: https://www.reddit.com/r/pdf/comments/1gcnpel/how_to_accurately_ocr_2_column_pages/

[^2_6]: https://blog.stackademic.com/nougat-ocr-extracting-information-from-academic-documents-has-never-been-easier-how-to-get-d1ffcbe2f62d

[^2_7]: https://github.com/jbarlow83/OCRmyPDF/issues/77

[^2_8]: https://stackoverflow.com/questions/31651071/how-to-ocr-multiple-column-in-a-document-using-tesseract

[^2_9]: https://facebookresearch.github.io/nougat/

[^2_10]: https://news.ycombinator.com/item?id=38482007

[^2_11]: https://pypi.org/project/marker-pdf/0.2.4/

[^2_12]: https://github.com/datalab-to/marker

[^2_13]: https://jimmysong.io/en/blog/pdf-to-markdown-open-source-deep-dive/

[^2_14]: https://adasci.org/a-practical-guide-to-text-generation-from-complex-pdfs-using-rag-with-llamaparse/

[^2_15]: https://www.llamaindex.ai/blog/introducing-llamacloud-and-llamaparse-af8cedf9006b

[^2_16]: https://www.youtube.com/watch?v=TYLUTIAn1Yg

[^2_17]: https://colab.research.google.com/github/KxSystems/kdbai-samples/blob/main/LlamaParse_pdf_RAG/llamaParse_demo.ipynb

[^2_18]: https://www.datacamp.com/tutorial/docling

[^2_19]: https://arxiv.org/html/2408.09869v1

[^2_20]: https://atalupadhyay.wordpress.com/2025/08/07/document-intelligence-guide-to-docling-for-ai-ready-data-processing/

[^2_21]: https://docling-project.github.io/docling/

[^2_22]: https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion

[^2_23]: https://docling-project.github.io/docling/usage/vision_models/

[^2_24]: https://unstructured.io/blog/mastering-pdf-transformation-strategies-with-unstructured-part-2

[^2_25]: https://unstructured.io/blog/mastering-table-extraction-revolutionize-your-earnings-reports-analysis-with-ai

[^2_26]: https://www.reddit.com/r/LangChain/comments/17c7g9b/anyone_worked_on_reading_pdf_with_tables/

[^2_27]: https://unstructured.io/blog/how-to-process-pdf-in-python

[^2_28]: https://arxiv.org/abs/2202.01414

[^2_29]: https://www.llamaindex.ai/blog/pdf-parsing-llamaparse

[^2_30]: https://github.com/facebookresearch/nougat

[^2_31]: https://ieeexplore.ieee.org/iel8/6287639/10820123/10845773.pdf

[^2_32]: https://www.reddit.com/r/LangChain/comments/1ef12q6/the_rag_engineers_guide_to_document_parsing/

[^2_33]: https://huggingface.co/spaces/ysharma/nougat/resolve/main/input/nougat.pdf

[^2_34]: https://www.reddit.com/r/LocalLLaMA/comments/192i8ew/ocr_techniques_for_rag_pdf_extraction/

[^2_35]: https://www.youtube.com/watch?v=NjQeS8Zu6yg

[^2_36]: https://www.youtube.com/watch?v=mdLBr9IMmgI

[^2_37]: https://github.com/docling-project/docling

[^2_38]: https://www.kaggle.com/code/ducnh279/parse-pdfs-marker-is-all-you-need/input

[^2_39]: https://towardsdatascience.com/docling-the-document-alchemist/

[^2_40]: https://github.com/Unstructured-IO/unstructured

[^2_41]: https://www.reddit.com/r/Rag/comments/1ilxf1i/best_pdf_parser_for_academic_papers/

