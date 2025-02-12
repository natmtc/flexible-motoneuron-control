# Neuromodulation increases the flexibility of motoneuron control

In this project we implement a biphysical model that builds upon Hodgkin-Huxley equations and that has two compartments - a dendrite and a soma. 

![alt text](pictures_for_explanation/circuit.png)

We checked the model could replicate a force-tracking task correctly by implementing a PD controller to minimizes the error between target and actual force.

The model is validated with experimental data from the Tibialis Anterior and Vastus Lateralis muscles of 8 participants (Avrillon et al. 2024, eLife) while they executed trapezoidal isometric contractions.

