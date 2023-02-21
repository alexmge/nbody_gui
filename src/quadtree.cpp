#include "quadtree.h"

Quadtree::Quadtree(sf::FloatRect bounds)
{
    this->body = NULL;
    this->bounds = bounds;
    this->northwest = NULL;
    this->northeast = NULL;
    this->southwest = NULL;
    this->southeast = NULL;
}

Quadtree::~Quadtree()
{
    if (this->northwest != NULL)
    {
        delete this->northwest;
    }
    if (this->northeast != NULL)
    {
        delete this->northeast;
    }
    if (this->southwest != NULL)
    {
        delete this->southwest;
    }
    if (this->southeast != NULL)
    {
        delete this->southeast;
    }
}

bool Quadtree::inBoundary(struct Vertex* body)
{
    return (body->position.x >= this->bounds.left && body->position.x <= this->bounds.left + this->bounds.width &&
        body->position.y >= this->bounds.top && body->position.y <= this->bounds.top + this->bounds.height);
}

int Quadtree::getQuadrant(struct Vertex* body)
{
    int quadrant = -1;
    float midX = this->bounds.left + this->bounds.width / 2;
    float midY = this->bounds.top + this->bounds.height / 2;
    bool top = body->position.y < midY && body->position.y >= this->bounds.top;
    bool bottom = body->position.y >= midY && body->position.y <= this->bounds.top + this->bounds.height;
    if (body->position.x < midX && body->position.x >= this->bounds.left)
    {
        if (top)
        {
            quadrant = 0;
        }
        else if (bottom)
        {
            quadrant = 2;
        }
    }
    else if (body->position.x >= midX && body->position.x <= this->bounds.left + this->bounds.width)
    {
        if (top)
        {
            quadrant = 1;
        }
        else if (bottom)
        {
            quadrant = 3;
        }
    }
    return quadrant;
}

void Quadtree::insert(struct Vertex *body)
{
    // if node is a leaf
    if (this->northwest == NULL)
    {
        // if node is empty
        if (this->body == NULL)
        {
            this->body = body;
            this->center_of_mass = body->position;
            this->total_mass = body->mass;
        }
        else
        {
            // if node is not empty
            // subdivide
            float midX = this->bounds.left + this->bounds.width / 2;
            float midY = this->bounds.top + this->bounds.height / 2;
            this->northwest = new Quadtree(sf::FloatRect(this->bounds.left, this->bounds.top, midX - this->bounds.left, midY - this->bounds.top));
            this->northeast = new Quadtree(sf::FloatRect(midX, this->bounds.top, this->bounds.left + this->bounds.width - midX, midY - this->bounds.top));
            this->southwest = new Quadtree(sf::FloatRect(this->bounds.left, midY, midX - this->bounds.left, this->bounds.top + this->bounds.height - midY));
            this->southeast = new Quadtree(sf::FloatRect(midX, midY, this->bounds.left + this->bounds.width - midX, this->bounds.top + this->bounds.height - midY));
            this->southeast->depth = this->depth + 1;
            this->southwest->depth = this->depth + 1;
            this->northeast->depth = this->depth + 1;
            this->northwest->depth = this->depth + 1;

            // insert old body
            int quadrant = this->getQuadrant(this->body);
            if (quadrant == 0 && this->northwest->depth < 11)
            {
                this->northwest->insert(this->body);
            }
            else if (quadrant == 1 && this->northeast->depth < 11)
            {
                this->northeast->insert(this->body);
            }
            else if (quadrant == 2 && this->southwest->depth < 11)
            {
                this->southwest->insert(this->body);
            }
            else if (quadrant == 3 && this->southeast->depth < 11)
            {
                this->southeast->insert(this->body);
            }

            this->body = NULL;

            // insert new body oui
            quadrant = this->getQuadrant(body);
            if (quadrant == 0 && this->northwest->depth < 11)
            {
                this->northwest->insert(body);
                this->center_of_mass = (this->center_of_mass * this->total_mass + body->position * body->mass) / (this->total_mass + body->mass);
                this->total_mass += body->mass;
            }
            else if (quadrant == 1 && this->northeast->depth < 11)
            {
                this->northeast->insert(body);
                this->center_of_mass = (this->center_of_mass * this->total_mass + body->position * body->mass) / (this->total_mass + body->mass);
                this->total_mass += body->mass;
            }
            else if (quadrant == 2 && this->southwest->depth < 11)
            {
                this->southwest->insert(body);
                this->center_of_mass = (this->center_of_mass * this->total_mass + body->position * body->mass) / (this->total_mass + body->mass);
                this->total_mass += body->mass;
            }
            else if (quadrant == 3 && this->southeast->depth < 11)
            {
                this->southeast->insert(body);
                this->center_of_mass = (this->center_of_mass * this->total_mass + body->position * body->mass) / (this->total_mass + body->mass);
                this->total_mass += body->mass;
            }
        }
    }
    else if (this->depth < 11) // if node is not a leaf
    {
        int quadrant = this->getQuadrant(body);
        if (quadrant == 0)
        {
            this->northwest->insert(body);
            this->center_of_mass = (this->center_of_mass * this->total_mass + body->position * body->mass) / (this->total_mass + body->mass);
            this->total_mass += body->mass;
        }
        else if (quadrant == 1)
        {
            this->northeast->insert(body);
            this->center_of_mass = (this->center_of_mass * this->total_mass + body->position * body->mass) / (this->total_mass + body->mass);
            this->total_mass += body->mass;
        }
        else if (quadrant == 2)
        {
            this->southwest->insert(body);
            this->center_of_mass = (this->center_of_mass * this->total_mass + body->position * body->mass) / (this->total_mass + body->mass);
            this->total_mass += body->mass;
        }
        else if (quadrant == 3)
        {
            this->southeast->insert(body);
            this->center_of_mass = (this->center_of_mass * this->total_mass + body->position * body->mass) / (this->total_mass + body->mass);
            this->total_mass += body->mass;
        }
    }
}

void Quadtree::draw(sf::RenderWindow& window)
{
    sf::RectangleShape rectangle;
    rectangle.setPosition(this->bounds.left, this->bounds.top);
    rectangle.setSize(sf::Vector2f(this->bounds.width, this->bounds.height));
    rectangle.setFillColor(sf::Color::Transparent);
    rectangle.setOutlineColor(sf::Color::White);
    rectangle.setOutlineThickness(1);
    window.draw(rectangle);
    if (this->northwest != NULL)
    {
        this->northwest->draw(window);
        this->northeast->draw(window);
        this->southwest->draw(window);
        this->southeast->draw(window);
    }
}

struct Vertex* Quadtree::getBody()
{
    return this->body;
}

sf::FloatRect Quadtree::getBounds()
{
    return this->bounds;
}

sf::Vector2f Quadtree::getCenterOfMass()
{
    return this->center_of_mass;
}

float Quadtree::getTotalMass()
{
    return this->total_mass;
}