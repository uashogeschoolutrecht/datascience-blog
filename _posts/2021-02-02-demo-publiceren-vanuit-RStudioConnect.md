---
layout: post
title:  "Hoe publiceer ik een data-product via RStudio?"
date:   2021-02-02 15:40:20 +0100
by: Marc Teunis
---

# Hoe publiceer ik een data-product via RStudio?
Door Marc Teunis

{% include rstudio-connect-video.html %}

Als onderzoeker wil je graag een idee krijgen van de patronen die in je data zitten. Wanneer je een experiment hebt gedaan of data hebt verzameld en je wilt (na flink wat moeite) je mooie analyse vastleggen en waarschijnlijk ook delen. Wanneer je een script-taal voor je analyse hebt gebruikt is er een makkelijke manier om de data-producten die je hebt gemaakt te delen. Ook het ontwikkelen van interactieve visualisaties of het bouwen van een dashboard is vaak een goede manier om jouw analyse en data te delen. Binnen Team DO ontwikkelen we tools om je als onderzoeker te helpen bij het uitvoeren van je analyses en het managen van je data. In deze demonstratie laat Marc Teunis zien hoe je vanuit RStudio een data-product zo kan publiceren dat jijzelf en anderen het data-product kunnen zien en met jouw data kan interacteren. Hij doet dit door middel van nieuwe software die binnen de Virtuele Onderzoeksomgeving beschikbaar is: RStudio::CONNECT. Kijk voor een sneak preview over dit product op: [https://rstudio.com/products/connect/](https://rstudio.com/products/connect/). Het delen van je analyse kan naar specifieke gebruikers en collega’s of openbaar voor iedereen, afhankelijk van jouw behoeften.

![screenshot](/blog/assets/demo-publiceren-vanuit-RStudioConnect/image-1.png)

## Links van de demo

- [Docs](https://datascience.hu.nl)
- [Aanvragen Workspaces](https://hogeschoolutrecht.topdesk.net/tas/public/ssp/content/serviceflow?unid=ebffb71962a94de68aa4f81ec25402fe)
- [RStudio Connect](https://datascience.hu.nl/rsconnect)
- [RMarkdown](https://bookdown.org/yihui/rmarkdown/)
- [Static COVID Example](https://datascience.hu.nl/rsconnect/covid_rmd/)
- [GitHub Source](https://github.com/uashogeschoolutrecht/covid_demo)
- [Shiny Example](https://datascience.hu.nl/rsconnect/covid-app)
- [Flipbook](https://datascience.hu.nl/rsconnect/connect/#/apps/27/access)

## Code walkthrough (in English for reproducibility reasons)
Coding examples, documentation and demos are mostly written in English. To align with the jargon and terms used in those demos, I am writing the demo below in English.

## Setup
To get started with publshing your data products form RStudio you need to set up a few things:
 
 1. Go to https://datascience.hu.nl/rsconnect/connect/ and click the get started button to register your account. This is the HU RStudio::CONNECT server that functions as a landing place for hosting you data products online
 2.Set up a Github account if you do not already have on. If you are serious about Data Science, you need a Github account. See [this blog]()
 3. An RStudio installation. You can apply for a HU RStudio workspace [here](https://hogeschoolutrecht.topdesk.net/tas/public/ssp/content/serviceflow?unid=ebffb71962a94de68aa4f81ec25402fe). If you have administrative right on your (HU) laptop, you can also install R and RStudio locally. If you are new to R and RStudio. [Here](https://education.rstudio.com/learn/beginner/) is a good starting point
 
## Ready?, Get set!, Go!! 
 
The ideal work flow is to first start with an empty github repository on github.com. Than we clone that repository to our RStudio environment. From RStudio we are going to create a simple RMarkdown file containing a short data analysis and some graphs. Once the RMarkdown is finished we will first render it locally to an html page. This way we can inspect the 'product' and check it's contents. Once we are satisfied, we will publish this product to the HU RStudio::CONNECT server. I will show you how you can than change the permission to view this webpage, so that you can share it with others.

## Create a new empty Github repo
The IT jargon for repository is repo, so if you want to sound like you know what you are talking about: use 'repo'. Once you login to your Github ccount, you can easily create a new github repo by clicking the green `New` button in your account repository home.
 
![create a new repo](/assets/demo-publiceren-vanuit-RStudioConnect/new_repo.png) 




