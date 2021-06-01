#include "Headers.h"


//--------------------------------------------------//
// Constructors and destructors.

Map::Map(){}

Map::Map(Generator* generator_ptr)
    : m_GeneratorPtr(generator_ptr){}

Map::~Map(){}


//--------------------------------------------------//
// Public methods.

void Map::paint(Graphics& g){
    g.fillAll(juce::Colours::black);
}

void Map::resized(){
    createSun();
}

void Map::createSun(){
    // Create sun object.
    m_Sun.add(new Sun(&m_Planets, m_GeneratorPtr));

    // Display sun.
    addAndMakeVisible(m_Sun[0]);

    // Draws sun to the center of the screen.
    m_Sun[0]->draw();
}


//--------------------------------------------------//
// Private methods.

void Map::createPlanet(int x, int y){
    Logger::writeToLog("\nCreating planet...");

    // Instantiate planet inside planets array.
    // Pointers to sun, planets and generator objects are passed.
    m_Planets.add(new Planet(&m_Planets, m_GeneratorPtr));
    
    // Update number of planets.
    m_NumPlanets = m_Planets.size();

    // Extra setup for planet object.
    setupPlanet(m_Planets[m_NumPlanets - 1], x, y);

    // Run latent mixture algorithm.
    mixLatents();
    
    Logger::writeToLog("Planet created.");
    Logger::writeToLog("Number of planets: " + std::to_string(m_NumPlanets) + "\n");
}

void Map::setPlanetID(Planet* planet){
    // Generate random ID for component.
    auto randomID = juce::String(juce::Random::getSystemRandom().nextInt(100001));    

    // Check if ID is unique.
    for(int i = 0; i < m_NumPlanets - 1; i++){
        if(planet->getComponentID() == randomID){
            while(planet->getComponentID() == randomID){
                randomID = juce::String(juce::Random::getSystemRandom().nextInt(100000)); 
            }
        }
    }
    
    // Set ID.
    planet->setComponentID(randomID);
}

void Map::setupPlanet(Planet* planet, int x, int y){
    setPlanetID(planet);

    addAndMakeVisible(planet);

    // Add listener for planet destruction request and lerp graph.
    planet->m_Destroy.addListener(this);
    planet->m_LerpGraph.addListener(this);

    // Render planet to screen.
    planet->setMapBoundaries(getWidth(), getHeight());
    
    planet->draw(
        planet->getDiameter(),
        x - (planet->getDiameter() / 2) - (planet->getClipBoundary() / 2),
        y - (planet->getDiameter() / 2) - (planet->getClipBoundary() / 2)
    );
    
}

void Map::destroyPlanet(){
    // For each planet check to see if it has been set for destruction.
    for(int i = 0; i < m_Planets.size(); i++){
        if(m_Planets[i]->m_Destroy == true){
            // Remove planet from array and delete.
            m_Planets.remove(i, true);

            // Reduce number of planets counter.
            m_NumPlanets -= 1;

            Logger::writeToLog("\nPlanet destroyed.\n");
        }
    }
}

float Map::getDistance(Sun* sun, Planet* planet){
    int centrePlanetX = planet->getCentreX(planet);
    int centrePlanetY = planet->getCentreY(planet);
    int centreSunX = sun->getX() + sun->getDiameter() / 2;
    int centreSunY = sun->getY() + sun->getDiameter() / 2;

    float a = (float)pow(centreSunX - centrePlanetX, 2.0f);
    float b = (float)pow(centreSunY - centrePlanetY, 2.0f);
    return sqrt(a + b);
}

float Map::getDistance(Planet* planet_a, Planet* planet_b){  
    int centreXA = planet_a->getCentreX(planet_a);
    int centreYA = planet_a->getCentreY(planet_a);
    int centreXB = planet_b->getCentreX(planet_b);
    int centreYB = planet_b->getCentreY(planet_b);

    float a = (float)pow(centreXB - centreXA, 2.0f);
    float b = (float)pow(centreYB - centreYA, 2.0f);

    return sqrt(a + b);
}

float Map::getForceVector(Sun* sun, Planet* planet){
    float r = getDistance(sun, planet);
    float m = ((float)sun->getDiameter() * (float)planet->getDiameter());
    
    Logger::writeToLog("Width: " + std::to_string(getWidth()));
    Logger::writeToLog("Distance: " + std::to_string(r));
    
    return (m / pow(r, 2.0f));
}

float Map::getForceVector(Planet* planet_a, Planet* planet_b){
    float r = getDistance(planet_a, planet_b);
    float m = ((float)planet_a->getDiameter() * (float)planet_b->getDiameter());
    
    Logger::writeToLog("Width: " + std::to_string(getWidth()));
    Logger::writeToLog("Distance: " + std::to_string(r));
    
    return (m / pow(r, 2.0f));
}

void Map::mixLatents(){
    Logger::writeToLog("Latents being mixed.");

    float forceVector;
    Planet* planet_a;
    Planet* planet_b;
    Sun* sun = m_Sun[0];
    sun->m_LerpLatents = sun->m_Latents;

    for(int i = 0; i < m_Planets.size(); i++){
        planet_a = m_Planets[i];
        planet_a->m_LerpLatents = planet_a->m_Latents;

        for(int j = 1; j < m_Planets.size(); j++){
            planet_b = m_Planets[j];
            if(planet_a->getComponentID() == planet_b->getComponentID()){continue;}
            
            forceVector = getForceVector(planet_a, planet_b);
            planet_a->m_LerpLatents = at::lerp(planet_a->m_LerpLatents, planet_b->m_Latents, forceVector);
        }
    }

    for(int i = 0; i < m_Planets.size(); i++){
        planet_a = m_Planets[i];

        forceVector = getForceVector(sun, planet_a);
        Logger::writeToLog("Force: " + std::to_string(forceVector));
        sun->m_LerpLatents = at::lerp(sun->m_LerpLatents, planet_a->m_LerpLatents, forceVector);
    }

    sun->generateSample(sun->m_LerpLatents);
}

void Map::mouseUp(const MouseEvent& e){}

void Map::mouseDoubleClick(const MouseEvent& e){
    Logger::writeToLog("Detected double click.");

    if(e.mods.isLeftButtonDown()){
        int eventX = e.getMouseDownX();
        int eventY = e.getMouseDownY();

        if(m_NumPlanets < M_MAX_NUM_PLANETS)
            createPlanet(eventX, eventY);
        else
            Logger::writeToLog("Maximum number of planets reached.");
    }
}

void Map::valueChanged(juce::Value& value){
    destroyPlanet();
    mixLatents();
}
