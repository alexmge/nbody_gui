#pragma once

#include "physics.cuh"

extern bool showQuadtree;
extern int screenWidth;
extern int screenHeight;
extern bool pause;
extern __device__ float gravity_constant;
extern bool barnesHut;

void GUI_Bodies(std::vector<Vertex>* bodies, Quadtree *tree, sf::Clock& deltaClock);
