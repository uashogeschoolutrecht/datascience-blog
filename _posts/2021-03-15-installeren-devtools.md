---
title: "Installeren van devtools op Linux"
layout: post
date: "2021-03-15 11:00:00 +0100"
by: Fraukje Coopmans
---

Door Fraukje Coopmans

***

Wil je graag een package installeren die niet direct van CRAN af komt? Bijvoorbeeld een package van GitHub?
Dan is het package `devtools` heel interessant om te installeren!

Als je R op Windows draait kun je deze simpelweg direct installeren via `install_packages('devtools')`, maar op Linux gaat je dit mogelijk een foutmelding opleveren. 

De oorzaak van deze foutmeldingen is dat er op de Rstudio omgeving het package `bspm` (Bridge to System Package Manager) standaard 'aan' staat. Dit package zorgt er voor dat je super snel nieuwe packages via precompiled binaries binnen kan halen. Echter, als er voor een package geen precompiled binary bestaat (in dit geval dus voor het package 'devtools') dan werkt de `install.packages()` niet. In dit geval moet je eerst het bspm package tijdelijk uitzetten via
`bspm::disable()` en dan kun je vervolgens via de normale route (`install_packages('devtools')`) je package installeren.

Vergeet achteraf de bspm niet weer the enablen zodat anderen installaties lekker snel gaan!
