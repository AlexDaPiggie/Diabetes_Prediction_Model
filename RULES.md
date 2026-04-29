# Machine Learning Agent Rules

## Code & Token Efficiency
- **Surgical Edits:** Only modify code blocks strictly necessary for the task. Do not refactor or rewrite functional existing code.
- **Incremental Logic:** When adding new logic, append to existing cells or create new ones rather than overwriting complex structures.
- **Concise Outputs:** Limit explanations to 2-3 sentences. Use bullet points for technical summaries.

## Data Science & Visualization
- **Feature Engineering:** Perform correlation analysis first. Examine features correlation sophisticately to combine features. Aggressively prune redundant or low-importance features.
- **Visualization:** Use **Seaborn** as the primary plotting library. Ensure plots are labeled and use a consistent color palette.
- **Analysis:** Place all interpretations, insights, and "Next Steps" in Markdown cells, not as code comments.
- **Examine:** Observe the graphs and the data carefully after running each cell. Implementing the code cells one by one. 

## Communication Style
- **Clarity:** Use plain, professional language. Avoid jargon unless it is a specific ML term.
- **Structure:** Use headers (##) in Markdown cells to separate different phases (e.g., Data Cleaning, Modeling).