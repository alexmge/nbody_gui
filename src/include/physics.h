#pragma once

#include <vector>
#include <cmath>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "quadtree.h"

void updateBodies(std::vector<Vertex>* bodies);
void updateBodies(std::vector<Vertex>* bodies, Quadtree* quadtree);