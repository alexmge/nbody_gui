#include "imgui.h"
#include "imgui-SFML.h"
#include "imgui-elements.cuh"

void QuadtreeTreeNode(Quadtree* tree)
{
    if (ImGui::TreeNode("Quadtree"))
    {
        if (tree->getBody() != nullptr)
        {
            ImGui::Text("Body: (%.2f, %.2f)", tree->getBody()->position.x, tree->getBody()->position.y);
        }
        else
        {
            ImGui::Text("Body: None");
        }
        ImGui::Text("Center of Mass: (%.2f, %.2f)", tree->getCenterOfMass().x, tree->getCenterOfMass().y);
        ImGui::Text("Total Mass: %.2f", tree->getTotalMass());
        if (tree->northwest != nullptr)
        {
            QuadtreeTreeNode(tree->northwest);
        }
        if (tree->northeast != nullptr)
        {
            QuadtreeTreeNode(tree->northeast);
        }
        if (tree->southwest != nullptr)
        {
            QuadtreeTreeNode(tree->southwest);
        }
        if (tree->southeast != nullptr)
        {
            QuadtreeTreeNode(tree->southeast);
        }
        ImGui::TreePop();
    }
}

void GUI_Bodies(std::vector<Vertex>* bodies, Quadtree *tree, sf::Clock& deltaClock)
{
    ImGui::Begin("Debug");
    // pause button
    if (ImGui::Button("Play/Pause"))
    {
        pause = !pause;
    }
    // checkbox for Barnes-Hut optimization
    ImGui::Checkbox("Barnes-Hut", &barnesHut);
    // number of fps
    ImGui::Text("FPS: %.2f", 1.f / deltaClock.restart().asSeconds());
    //number of bodies
    ImGui::Text("Number of bodies: %lu", bodies->size());
    //slider to change the value of gravity
    ImGui::SliderFloat("Gravity", &gravity_constant, -1, 1);
    if (ImGui::TreeNode("Bodies Options"))
    {
        if (ImGui::Button("Add Body"))
        {
            Vertex v;
            v.position = Vector(rand() % screenWidth, rand() % screenHeight);
            v.acceleration = Vector(0, 0);
            v.velocity = Vector(0, 0);
            v.mass = 1;
            v.color = sf::Color::White;
            bodies->push_back(v);
        }
        if (ImGui::Button("Add 1000 Bodies disk"))
{
    float center_x = screenWidth/2;
    float center_y = screenHeight/2;
    // add the points so that they are evenly distributed on a plain disk
    for (int i = 0; i < 1000; i++)
    {
        Vertex v;
        float angle = 2 * M_PI * i / 1000;
        v.position = Vector(center_x + 100 * cos(angle), center_y + 100 * sin(angle));
        v.acceleration = Vector(0, 0);
        // tangential velocity
        v.velocity = Vector(-100 * sin(angle) * 0.01f, 100 * cos(angle) * 0.01f);
        v.mass = 1;
        v.color = sf::Color::White;
        bodies->push_back(v);
    }
}

        if (ImGui::Button("Remove Body"))
        {
            if (bodies->size() > 0)
            {
                bodies->pop_back();
            }
        }
        if (ImGui::Button("Clear Bodies"))
        {
            bodies->clear();
        }
        if (ImGui::TreeNode("See properties"))
        {
            // make a treenode for each body
            for (size_t i = 0; i < bodies->size(); i++)
            {
                if (ImGui::TreeNode(("Body " + std::to_string(i)).c_str()))
                {
                    ImGui::Text("Position: (%.2f, %.2f)", bodies->at(i).position.x, bodies->at(i).position.y);
                    ImGui::Text("Velocity: (%.2f, %.2f)", bodies->at(i).velocity.x, bodies->at(i).velocity.y);
                    ImGui::Text("Acceleration: (%.2f, %.2f)", bodies->at(i).acceleration.x, bodies->at(i).acceleration.y);
                    ImGui::Text("Mass: %.2f", bodies->at(i).mass);
                    ImGui::Text("Color: (%d, %d, %d)", bodies->at(i).color.r, bodies->at(i).color.g, bodies->at(i).color.b);
                    ImGui::TreePop();
                }
            }
            ImGui::TreePop();
        }
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Quadtree Options"))
    {
        ImGui::Checkbox("Show Quadtree", &showQuadtree);
        QuadtreeTreeNode(tree);
        ImGui::TreePop();
    }
    ImGui::End();
}