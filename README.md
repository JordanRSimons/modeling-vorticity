# Vorticity Modeling and Analysis

When waves approach the coastline, they can create complex circulating current patterns that trap ocean wildlife. Quantifying these circulation patterns with a metric called vorticity could benefit both oceanographers and ecologists. Scientists have recently developed a tool to compute vorticity by tracking the movements of sensor buoys placed in the water (Harms et al. 2023). The goal of this project was to benchmark this new model against more established ones in order to minimize the number of buoys required to get accurate results.

The animation below shows an example of simulated buoys, also called test particles, following a circulating current. The white peninsulas represent land and the waves originate from the top of the plot.

![Particle Animation](images/hour.gif)
*An animation tracking 1200 simulated test particles (buoys) along a coastline over an hour. The off-white color represents land ad the colored regions depict vorticity. See [notebook 1](https://github.com/JordanRSimons/modeling-vorticity/blob/main/notebooks/01_visualizations_and_results.md) for details.*

# Repo Contents

This repository contains an overview of Python modeling and data visualization done during a summer research project in applied physics and oceanography at the University of Washington.

* ['notebooks/01_visualizations_and_results'](https://github.com/JordanRSimons/modeling-vorticity/blob/main/notebooks/01_visualizations_and_results.md): A discussion of my results and a showcase of plots and animations and the code that produced them
* ['notebooks/02_generate_particle_data.md](https://github.com/JordanRSimons/modeling-vorticity/blob/main/notebooks/02_generate_particle_data.md): A technical explanation of my approach to modeling vorticity
* ['images/'](https://github.com/JordanRSimons/modeling-vorticity/tree/main/images): A directory with all plots and animations as standalone files
* ['poster/JordanCICOESPoster.pdf'](https://github.com/JordanRSimons/modeling-vorticity/blob/main/poster/JordanCICOESPoster.pdf): My research poster, created in Microsoft PowerPoint
* [YouTube Video](https://www.youtube.com/watch?v=HQ3oBR151bI): A lighthearted 2-minute video introduction to the project intended for the general public

# Important Note

This project illustrates my skills in modeling, data visualization, and scientific storytelling. Given the size of the datasets and the proprietary nature of the models, this repository does not include fully runnable code.

This research was supervised and mentored by Dr. Walter Torres and was supported by a US National Science Foundation “Research Experiences for Undergraduates” grant to the Cooperative Institute for Climate, Ocean, & Ecosystem Studies. 




