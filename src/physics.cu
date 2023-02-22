#include <iostream>
#include <stack>
#include "physics.cuh"
#include <thread>

#define SOFTENING 2

__device__ float gravity_constant = 0.1;

/**
 * @brief compute the force between two bodies according to newton's law of universal gravitation
 * 
 * @param body1 the first body
 * @param body2 the second body
 * 
 * @return Vector the force between the two bodies
 */
__device__
void computeForce(Vertex* body1, Vertex* body2, Vector* force)
{
    // compute the distance between the two bodies
    Vector distance = Vector(0, 0);
    distance.x = body2->position.x - body1->position.x;
    distance.y = body2->position.y - body1->position.y;
    float distanceMagnitude = sqrt(pow(distance.x, 2) + pow(distance.y, 2));

    // compute the force between the two bodies
    float forceMagnitude =  gravity_constant * (body1->mass * body2->mass) / (pow(distanceMagnitude, 2) + 10);
    Vector res = distance * forceMagnitude / distanceMagnitude;

    *force += res;
}

/**
 * @brief compute the force between a body and a quadtree according to newton's law of universal gravitation
 * 
 * @param body the body
 * @param quadtree the quadtree
 * 
 * @return Vector the force between the body and the quadtree
 */
void computeForce_qtree(Vertex* body, Quadtree* quadtree, Vector* force)
{
    // Stack to store the quadtree nodes that need to be processed
    std::stack<Quadtree*> nodes;

    // Push the root node onto the stack
    nodes.push(quadtree);

    Vector res = Vector(0, 0);

    while (!nodes.empty())
    {
        Quadtree* node = nodes.top();
        nodes.pop();

        // If the node is a leaf node and it is not the body itself, add the force
        // between the body and the leaf node to the total force
        if (node->getBody() != NULL && node->getBody() != body)
        {
            Vector distance = node->getBody()->position - body->position;
            float distanceMagnitude = sqrt(pow(distance.x, 2) + pow(distance.y, 2));
            float forceMagnitude = gravity_constant * (body->mass * node->getTotalMass()) / (pow(distanceMagnitude, 2) + SOFTENING);

            res += distance * forceMagnitude / (distanceMagnitude + SOFTENING);
        }
        // Otherwise, calculate the ratio s/d (current region's width / distance to center of mass)
        else
        {
            // Compute the distance between the body and the center of mass
            Vector distance = node->getCenterOfMass() - body->position;
            float distanceMagnitude = sqrt(pow(distance.x, 2) + pow(distance.y, 2));

            // Compute the ratio s/d
            float ratio = node->getBounds().width / (distanceMagnitude + SOFTENING);

            // If s/d < theta, treat the current node as a single body, and calculate the force it exerts on body
            // Otherwise, push the children of the current node onto the stack to be processed later
            if (ratio < 0.9)
            {
                float forceMagnitude =  gravity_constant * (body->mass * node->getTotalMass()) / (pow(distanceMagnitude, 2) + SOFTENING);

                res += distance * forceMagnitude / (distanceMagnitude + SOFTENING);
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

    *force += res;
}

/**
 * @brief update the position, velocity and acceleration of all the bodies in the bodies vector
 * 
 * @param bodies the vector of bodies
 */
__global__
void updateBodies(Vertex* bodies, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // for each body, compute the force it exerts on all the other bodies
    for (int i = index; i < size; i += stride)
    {
        Vertex* body1 = &bodies[i];
        
        for (int j = 0; j < size; j++)
        {
            if (i != j)
            {
                Vertex* body2 = &bodies[j];
                Vector force(0, 0);
                computeForce(body1, body2, &force);
                body1->acceleration += force / body1->mass;
            }
        }
    }

    // update the position, velocity and acceleration of each body
    for (int i = index; i < size; i += stride)
    {
        Vertex* body = &bodies[i];
        body->velocity += body->acceleration;
        body->position += body->velocity;
        body->acceleration = Vector(0, 0);
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
    // for each body, compute the force it exerts on all the other bodies
    for (Vertex& body : *bodies)
    {
        Vector force(0, 0);
        computeForce_qtree(&body, quadtree, &force);
        body.acceleration += force / body.mass;
    }

    // update the position, velocity and acceleration of each body
    for (Vertex& body : *bodies)
    {
        body.velocity += body.acceleration;
        body.position += body.velocity;
        body.acceleration = Vector(0, 0);
    }
}
