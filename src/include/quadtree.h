#include <SFML/System.hpp>
#include <SFML/Graphics.hpp>
#include "vector.cuh"


struct Vertex
{
    Vector position;
    Vector velocity;
    Vector acceleration;
    float mass;
    sf::Color color;
};

class Quadtree
{
public:
    int getQuadrant(struct Vertex* body);
    int depth;
    float total_mass;
    struct Vertex *body;
    sf::FloatRect bounds;
    Vector center_of_mass;
    Quadtree(sf::FloatRect bounds);
    ~Quadtree();
    void insert(struct Vertex *body);
    void draw(sf::RenderWindow& window);
    struct Quadtree *northwest;
    struct Quadtree *northeast;
    struct Quadtree *southwest;
    struct Quadtree *southeast;
    struct Vertex* getBody();
    sf::FloatRect getBounds();
    Vector getCenterOfMass();
    float getTotalMass();
    bool inBoundary(struct Vertex* body);
};