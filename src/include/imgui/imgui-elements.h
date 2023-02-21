#pragma once

#include "physics.h"

extern bool showQuadtree;
extern int screenWidth;
extern int screenHeight;
extern bool pause;
extern float gravity_constant;
extern bool barnesHut;

void GUI_Bodies(std::vector<Vertex>* bodies, Quadtree *tree, sf::Clock& deltaClock);
