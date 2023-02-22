#pragma once

#include <vector>
#include <cmath>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "quadtree.h"

__global__ void updateBodies(Vertex* bodies, int size);
void updateBodies(std::vector<Vertex>* bodies, Quadtree* quadtree);
