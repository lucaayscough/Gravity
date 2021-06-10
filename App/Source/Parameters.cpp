#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

Parameters::Parameters(juce::ValueTree v)
    :   rootNode(v){
    Logger::writeToLog("Parameters created!");

    // Listeners.
    rootNode.addListener(this);
    updateMap.setValue(false);

    // Basic elements.
    addSunNode();
    juce::ValueTree rootPlanetNode(rootPlanetType);
    rootNode.addChild(rootPlanetNode, -1, nullptr);
}

Parameters::~Parameters(){
    rootNode.removeListener(this);
    Logger::writeToLog("Parameters destroyed!");
}

//------------------------------------------------------------//
// Structuring methods.

void Parameters::addSunNode(){
    juce::ValueTree sunNode(sunType);
    sunNode.setProperty(diameterProp, Variables::SUN_DIAMETER, nullptr);
    sunNode.setProperty(idProp, SUN_ID, nullptr);

    // Listeners.
    sunNode.setProperty(updateGraphSignal, false, nullptr);
    sunNode.setProperty(generateSampleSignal, false, nullptr);
    sunNode.addListener(this);

    // Sample.
    generateNewSample(sunNode);
    rootNode.addChild(sunNode, -1, nullptr);
    setActivePlanet(sunNode);
}

void Parameters::addPlanetNode(){
    juce::ValueTree planetNode(planetType);
    setRandomID(planetNode);
    planetNode.setProperty(diameterProp, Variables::DEFAULT_PLANET_DIAMETER, nullptr);

    // Listeners.
    planetNode.setProperty(updateGraphSignal, false, nullptr);
    planetNode.setProperty(generateSampleSignal, false, nullptr);
    
    // Sample.
    generateNewSample(planetNode);
    getRootPlanetNode().addChild(planetNode, -1, nullptr);
}

void Parameters::removePlanetNode(const juce::String& id){
    for(int i = 0; i < getRootPlanetNode().getNumChildren(); i++){
        if(getRootPlanetNode().getChild(i).getProperty(idProp) == id){
            getRootPlanetNode().removeChild(i, nullptr);
        }
    }
}

void Parameters::clearSamples(juce::ValueTree node){
    // TODO:
    // Cleanup this function.

    for(int i = 0; i < node.getChildWithName(rootPlanetType).getNumChildren(); i++){
        node.getChildWithName(rootPlanetType).getChild(i).removeProperty(latentsProp, nullptr);
        node.getChildWithName(rootPlanetType).getChild(i).removeProperty(lerpLatentsProp, nullptr);
        node.getChildWithName(rootPlanetType).getChild(i).removeProperty(sampleProp, nullptr);
    }
    
    node.getChildWithName(sunType).removeProperty(latentsProp, nullptr);
    node.getChildWithName(sunType).removeProperty(lerpLatentsProp, nullptr);
    node.getChildWithName(sunType).removeProperty(sampleProp, nullptr);
}

void Parameters::rebuildSamples(){
    // TODO:
    // Cleanup this function.

    for(int i = 0; i < getRootPlanetNode().getNumChildren(); i++){
        generateLatents(getRootPlanetNode().getChild(i));
        generateLerpLatents(getRootPlanetNode().getChild(i));
        generateSample(getRootPlanetNode().getChild(i), getLatents(getRootPlanetNode().getChild(i), latentsProp));
    }

    generateLatents(rootNode.getChildWithName(sunType));
    generateLerpLatents(rootNode.getChildWithName(sunType));
    generateSample(rootNode.getChildWithName(sunType), getLatents(rootNode.getChildWithName(sunType), latentsProp));

    mixLatents();
}

//------------------------------------------------------------//
// Tensor operations.

void Parameters::generateLatents(juce::ValueTree node){
    ReferenceCountedTensor::Ptr latents = new ReferenceCountedTensor(Generator::generateLatents(getSeed(node)));
    node.setProperty(latentsProp, juce::var(latents), nullptr);
}

void Parameters::generateLerpLatents(juce::ValueTree node){
    ReferenceCountedTensor::Ptr lerpLatents = new ReferenceCountedTensor(getLatents(node, latentsProp));
    node.setProperty(lerpLatentsProp, juce::var(lerpLatents), nullptr);
}

void Parameters::generateSample(juce::ValueTree node, at::Tensor tensor){
    node.setProperty(sampleProp, Generator::generateSample(tensor), nullptr);
}

void Parameters::generateNewSample(juce::ValueTree node){
    setRandomSeed(node);
    generateLatents(node);
    generateLerpLatents(node);
    generateSample(node, getLatents(node, latentsProp));
}

void Parameters::mixLatents(){
    float forceVector;

    generateLerpLatents(getSunNode());

    for(int i = 0; i < getRootPlanetNode().getNumChildren(); i++){
        auto planet_a = getRootPlanetNode().getChild(i);
        generateLerpLatents(planet_a);

        for(int j = 0; j < getRootPlanetNode().getNumChildren(); j++){
            if(i == j){continue;}

            auto planet_b = getRootPlanetNode().getChild(j);
            if(getID(planet_a) == getID(planet_b)){continue;}
            
            forceVector = getForceVector(planet_a, planet_b);
            at::Tensor newLatents = at::lerp(getLatents(planet_a, lerpLatentsProp), getLatents(planet_b, latentsProp), forceVector);
            setLatents(planet_a, lerpLatentsProp, newLatents);
        }
    }

    for(int i = 0; i < getRootPlanetNode().getNumChildren(); i++){
        auto planet = getRootPlanetNode().getChild(i);

        forceVector = getForceVector(getSunNode(), planet);
        at::Tensor newLatents = at::lerp(getLatents(getSunNode(), lerpLatentsProp), getLatents(planet, lerpLatentsProp), forceVector);
        setLatents(getSunNode(), lerpLatentsProp, newLatents);
    }

    generateSample(getSunNode(), getLatents(getSunNode(), lerpLatentsProp));
}

//------------------------------------------------------------//
// Get methods.

juce::ValueTree Parameters::getActivePlanet(){
    if(getSunNode().getProperty(isActiveProp)){return getSunNode();}
    for(int i = 0; i < getRootPlanetNode().getNumChildren(); i++){
        if(getRootPlanetNode().getChild(i).getProperty(isActiveProp)){return getRootPlanetNode().getChild(i);}
    }
}

juce::ValueTree Parameters::getSunNode(){return rootNode.getChildWithName(sunType);}
juce::ValueTree Parameters::getRootPlanetNode(){return rootNode.getChildWithName(rootPlanetType);}
std::int64_t Parameters::getSeed(juce::ValueTree node){return node.getProperty(seedProp);}
at::Tensor Parameters::getLatents(juce::ValueTree node, juce::Identifier& id){return ((ReferenceCountedTensor*)node.getProperty(id).getObject())->getTensor();}
juce::String Parameters::getID(juce::ValueTree node){return node.getProperty(idProp);}

float Parameters::getDistance(juce::ValueTree node_a, juce::ValueTree node_b){  
    int centreXA = node_a.getProperty(posCentreXProp);
    int centreYA = node_a.getProperty(posCentreYProp);
    int centreXB = node_b.getProperty(posCentreXProp);
    int centreYB = node_b.getProperty(posCentreYProp);

    float a = (float)pow(centreXB - centreXA, 2.0f);
    float b = (float)pow(centreYB - centreYA, 2.0f);

    return sqrt(a + b);
}

float Parameters::getForceVector(juce::ValueTree node_a, juce::ValueTree node_b){
    float r = getDistance(node_a, node_b);
    float m = ((float)node_a.getProperty(diameterProp) * (float)node_b.getProperty(diameterProp));
    return (m / pow(r, 2.0f));
}

//------------------------------------------------------------//
// Set methods.

void Parameters::setActivePlanet(juce::ValueTree node){
    for(int i = 0; i < getRootPlanetNode().getNumChildren(); i++){getRootPlanetNode().getChild(i).setProperty(isActiveProp, false, nullptr);}
    getSunNode().setProperty(isActiveProp, false, nullptr);

    node.setProperty(isActiveProp, true, nullptr);
}

void Parameters::setRandomID(juce::ValueTree node){
    // Generate random ID for component.
    auto randomID = juce::String(juce::Random::getSystemRandom().nextInt(1000));    

    // Check if ID is unique.
    for(int i = 0; i < getRootPlanetNode().getNumChildren() - 1; i++){
        if(getRootPlanetNode().getChild(i).getProperty(idProp) == randomID){
            while(getRootPlanetNode().getChild(i).getProperty(idProp) == randomID){
                randomID = juce::String(juce::Random::getSystemRandom().nextInt(1000)); 
            }
        }
    }

    // Set ID.
    node.setProperty(idProp, randomID, nullptr);
}

void Parameters::setRandomSeed(juce::ValueTree node){
    std::int64_t seed = juce::Random::getSystemRandom().nextInt64();
    if(seed < 0){seed = seed * -1;}
    node.setProperty(seedProp, seed, nullptr);
}

void Parameters::setLatents(juce::ValueTree node, juce::Identifier& id, at::Tensor& latents){
    ReferenceCountedTensor::Ptr lerpLatents = new ReferenceCountedTensor(latents);
    node.setProperty(lerpLatentsProp, juce::var(lerpLatents), nullptr);
    node.setProperty(id, juce::var(lerpLatents), nullptr);
}

//------------------------------------------------------------//
// Callback methods.

void Parameters::valueTreePropertyChanged(juce::ValueTree& node, const juce::Identifier& id){
    if(id == generateSampleSignal){
        if((bool)node.getProperty(id) == true){
            generateNewSample(node);
            node.setProperty(id, false, nullptr);
            Logger::writeToLog("New sample.");
        }
    }

    if(id == updateGraphSignal){
        if((bool)node.getProperty(id) == true){
            mixLatents();
            node.setProperty(id, false, nullptr);
            Logger::writeToLog("Update graph.");
        }
    }

    if(id == isActiveProp){
        
    }
}

//------------------------------------------------------------//
// Type identifiers.

juce::Identifier Parameters::sunType("Sun");
juce::Identifier Parameters::rootPlanetType("Root_Planet");
juce::Identifier Parameters::planetType("Planet");

//------------------------------------------------------------//
// Property identifiers.

juce::Identifier Parameters::idProp("ID");
juce::Identifier Parameters::isActiveProp("Is_Active");
juce::Identifier Parameters::mapWidthProp("Map_Width");
juce::Identifier Parameters::mapHeightProp("Map_Height");
juce::Identifier Parameters::diameterProp("Diameter");
juce::Identifier Parameters::posXProp("Position_X");
juce::Identifier Parameters::posYProp("Position_Y");
juce::Identifier Parameters::posCentreXProp("Position_Centre_X");
juce::Identifier Parameters::posCentreYProp("Position_Centre_Y");
juce::Identifier Parameters::colourProp("Colour");
juce::Identifier Parameters::seedProp("Seed");
juce::Identifier Parameters::latentsProp("Latents");
juce::Identifier Parameters::lerpLatentsProp("Interpolated_Latents");
juce::Identifier Parameters::sampleProp("Sample");

//------------------------------------------------------------//
// Callback signalers.

juce::Identifier Parameters::updateGraphSignal("Update_Graph");
juce::Identifier Parameters::generateSampleSignal("Generate_Sample");

