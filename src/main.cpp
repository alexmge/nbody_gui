#include "imgui.h"
#include "imgui-SFML.h"
#include "imgui-elements.h"
#include "physics.h"


bool showQuadtree = false;
bool barnesHut = false;
bool pause = false;
int screenWidth = 1920;
int screenHeight = 1080;

/**
 * @brief draw all the bodies in the bodies vector
 */
void drawbodies(sf::RenderWindow& window, std::vector<Vertex>* bodies)
{
    for (Vertex& v : *bodies)
    {
        sf::CircleShape shape(1.f);
        shape.setFillColor(v.color);
        shape.setPosition(v.position);
        window.draw(shape);
    }
}

/**
 * @brief Insert all the bodies into the quadtree
 */
void insertBodies(Quadtree* tree, std::vector<Vertex>* bodies)
{
    for (Vertex& v : *bodies)
    {
        tree->insert(&v);
    }
}

int main(void)
{
    // fullscreen
    sf::RenderWindow window(sf::VideoMode(screenWidth, screenHeight), "N-Body Simulation");
    window.setFramerateLimit(144);
    
    if (!ImGui::SFML::Init(window))
        return 1;
        
    std::vector<Vertex>* bodies = new std::vector<Vertex>();

    sf::Clock deltaClock;

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            ImGui::SFML::ProcessEvent(event);

            if (event.type == sf::Event::Closed)
            {
                window.close();
            }
        }

        ImGui::SFML::Update(window, deltaClock.restart());

        Quadtree* tree = new Quadtree(sf::FloatRect(0, 0, screenWidth, screenHeight));
        tree->depth = 0;

        insertBodies(tree, bodies);

        if (!pause)
        {
            if (barnesHut)
            {
                updateBodies(bodies, tree);
            }
            else
            {
                updateBodies(bodies);
            }
        }

        window.clear(sf::Color::Black);

        drawbodies(window, bodies);

        if (showQuadtree)
        {
            tree->draw(window);
        }

        GUI_Bodies(bodies, tree, deltaClock);

        ImGui::SFML::Render(window);
        window.display();
        delete tree;
    }

    delete bodies;

    ImGui::SFML::Shutdown();

    return 0;
}