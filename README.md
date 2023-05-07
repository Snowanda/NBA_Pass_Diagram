# Draw NBA pass diagram using plotly and networkx circular layout with directions visualized
A homework from my university data science course, practicing to draw diagrams from dataframes using Python plotly and networkx circular_layout.
![2021-22-GSW](2021-22-GSW.png)
As shown in the diagram, edges(passes) are thicker for edges with higher pass count, color ranging from blue to red reprsents pass quality based on the passed to players' total FGM/FGA compared with the FGM/FGA after this pass. Edges with less than 100 pass count are hidden so the diagram looks cleaner.
