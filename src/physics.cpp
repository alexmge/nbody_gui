#include <iostream>
#include <stack>
#include "physics.h"
#include <thread>

#define SOFTENING 2

float gravity_constant = 0.1;

/**
 * @brief compute the force between two bodies according to newton's law of universal gravitation
 * 
 * @param body1 the first body
 * @param body2 the second body
 * 
 * @return sf::Vector2f the force between the two bodies
 */
sf::Vector2f computeForce(Vertex* body1, Vertex* body2)
{
    // compute the distance between the two bodies
    sf::Vector2f distance = body2->position - body1->position;
    float distanceMagnitude = sqrt(pow(distance.x, 2) + pow(distance.y, 2));

    // compute the force between the two bodies
    float forceMagnitude =  gravity_constant * (body1->mass * body2->mass) / (pow(distanceMagnitude, 2) + 10);
    sf::Vector2f force = forceMagnitude * distance / distanceMagnitude;

    return force;
}

/**
 * @brief compute the force between a body and a quadtree according to newton's law of universal gravitation
 * 
 * @param body the body
 * @param quadtree the quadtree
 * 
 * @return sf::Vector2f the force between the body and the quadtree
 */
sf::Vector2f computeForce(Vertex* body, Quadtree* quadtree)
{
    // Stack to store the quadtree nodes that need to be processed
    std::stack<Quadtree*> nodes;

    // Push the root node onto the stack
    nodes.push(quadtree);

    sf::Vector2f force(0, 0);

    while (!nodes.empty())
    {
        Quadtree* node = nodes.top();
        nodes.pop();

        // If the node is a leaf node and it is not the body itself, add the force
        // between the body and the leaf node to the total force
        if (node->getBody() != NULL && node->getBody() != body)
        {
            sf::Vector2f distance = node->getBody()->position - body->position;
            float distanceMagnitude = sqrt(pow(distance.x, 2) + pow(distance.y, 2));
            float forceMagnitude = gravity_constant * (body->mass * node->getTotalMass()) / (pow(distanceMagnitude, 2) + SOFTENING);

            force += forceMagnitude * distance / (distanceMagnitude + SOFTENING);
        }
        // Otherwise, calculate the ratio s/d (current region's width / distance to center of mass)
        else
        {
            // Compute the distance between the body and the center of mass
            sf::Vector2f distance = node->getCenterOfMass() - body->position;
            float distanceMagnitude = sqrt(pow(distance.x, 2) + pow(distance.y, 2));

            // Compute the ratio s/d
            float ratio = node->getBounds().width / (distanceMagnitude + SOFTENING);

            // If s/d < theta, treat the current node as a single body, and calculate the force it exerts on body
            // Otherwise, push the children of the current node onto the stack to be processed later
            if (ratio < 0.9)
            {
                float forceMagnitude =  gravity_constant * (body->mass * node->getTotalMass()) / (pow(distanceMagnitude, 2) + SOFTENING);

                force += forceMagnitude * distance / (distanceMagnitude + SOFTENING);
            }
            else
            {
                if (node->northwest != NULL)
                {
                    nodes.push(node->northwest);
                }
                if (node->northeast != NULL)
                {
                    nodes.push(node->northeast);
                }
                if (node->southwest != NULL)
                {
                    nodes.push(node->southwest);
                }
                if (node->southeast != NULL)
                {
                    nodes.push(node->southeast);
                }
            }
        }
    }

    return force;
}

/**
 * @brief update the position, velocity and acceleration of all the bodies in the bodies vector
 * 
 * @param bodies the vector of bodies
 */
void updateBodies(std::vector<Vertex>* bodies)
{
    // use two for loops to compute the force between each body
    for (Vertex& body1 : *bodies)
    {
        for (Vertex& body2 : *bodies)
        {
            if (&body1 != &body2)
            {
                sf::Vector2f force = computeForce(&body1, &body2);
                body1.acceleration += force / body1.mass;
            }
        }
    }

    // update the position, velocity and acceleration of each body
    for (Vertex& body : *bodies)
    {
        body.velocity += body.acceleration;
        body.position += body.velocity;
        body.acceleration = sf::Vector2f(0, 0);
    }
}

/**
 * @brief update the position, velocity and acceleration of all the bodies in the bodies vector using the quadtree
 * 
 * @param bodies the vector of bodies
 * @param quadtree the quadtree
 */
void updateBodies(std::vector<Vertex>* bodies, Quadtree* quadtree)
{
    const size_t num_threads = std::min(bodies->size(), 15ul) + 1;
    const size_t block_size = (bodies->size() + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i)
    {
        const size_t start = i * block_size;
        const size_t end = std::min(start + block_size, bodies->size());

        threads.push_back(std::thread([&, start, end](){
            for (size_t j = start; j < end; ++j)
            {
                Vertex& body = (*bodies)[j];
                sf::Vector2f force = computeForce(&body, quadtree);
                body.acceleration += force / body.mass;
            }
        }));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    threads.clear();
    for (size_t i = 0; i < num_threads; ++i)
    {
        const size_t start = i * block_size;
        const size_t end = std::min(start + block_size, bodies->size());

        threads.push_back(std::thread([&, start, end](){
            for (size_t j = start; j < end; ++j)
            {
                Vertex& body = (*bodies)[j];
                body.velocity += body.acceleration;
                body.position += body.velocity;
                body.acceleration = sf::Vector2f(0, 0);
            }
        }));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }
}

