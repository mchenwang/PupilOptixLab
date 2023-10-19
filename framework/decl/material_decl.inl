// clang-format off
// #define PUPIL_MATERIAL_TYPE_DEFINE(type)
#ifdef PUPIL_MATERIAL_TYPE_DEFINE
PUPIL_MATERIAL_TYPE_DEFINE(Diffuse)
PUPIL_MATERIAL_TYPE_DEFINE(Dielectric)
PUPIL_MATERIAL_TYPE_DEFINE(RoughDielectric)
PUPIL_MATERIAL_TYPE_DEFINE(Conductor)
PUPIL_MATERIAL_TYPE_DEFINE(RoughConductor)
PUPIL_MATERIAL_TYPE_DEFINE(Plastic)
PUPIL_MATERIAL_TYPE_DEFINE(RoughPlastic)
PUPIL_MATERIAL_TYPE_DEFINE(Hair)
#endif

#ifdef PUPIL_MATERIAL_ATTR_DEFINE
PUPIL_MATERIAL_ATTR_DEFINE(diffuse)
PUPIL_MATERIAL_ATTR_DEFINE(dielectric)
PUPIL_MATERIAL_ATTR_DEFINE(rough_dielectric)
PUPIL_MATERIAL_ATTR_DEFINE(conductor)
PUPIL_MATERIAL_ATTR_DEFINE(rough_conductor)
PUPIL_MATERIAL_ATTR_DEFINE(plastic)
PUPIL_MATERIAL_ATTR_DEFINE(rough_plastic)
PUPIL_MATERIAL_ATTR_DEFINE(hair)
#endif

#ifdef PUPIL_MATERIAL_NAME_DEFINE
PUPIL_MATERIAL_NAME_DEFINE(diffuse)
PUPIL_MATERIAL_NAME_DEFINE(dielectric)
PUPIL_MATERIAL_NAME_DEFINE(roughdielectric)
PUPIL_MATERIAL_NAME_DEFINE(conductor)
PUPIL_MATERIAL_NAME_DEFINE(roughconductor)
PUPIL_MATERIAL_NAME_DEFINE(plastic)
PUPIL_MATERIAL_NAME_DEFINE(roughplastic)
PUPIL_MATERIAL_NAME_DEFINE(hair)
#endif

#ifdef PUPIL_MATERIAL_TYPE_ATTR_DEFINE
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(Diffuse,         diffuse)
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(Dielectric,      dielectric)
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(RoughDielectric, rough_dielectric)
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(Conductor,       conductor)
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(RoughConductor,  rough_conductor)
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(Plastic,         plastic)
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(RoughPlastic,    rough_plastic)
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(Hair,            hair)
#endif

#ifdef PUPIL_MATERIAL_ALBEDO_DEFINE
PUPIL_MATERIAL_ALBEDO_DEFINE(Diffuse,         diffuse,          reflectance)
PUPIL_MATERIAL_ALBEDO_DEFINE(Dielectric,      dielectric,       specular_reflectance)
PUPIL_MATERIAL_ALBEDO_DEFINE(RoughDielectric, rough_dielectric, specular_reflectance)
PUPIL_MATERIAL_ALBEDO_DEFINE(Conductor,       conductor,        specular_reflectance)
PUPIL_MATERIAL_ALBEDO_DEFINE(RoughConductor,  rough_conductor,  specular_reflectance)
PUPIL_MATERIAL_ALBEDO_DEFINE(Plastic,         plastic,          diffuse_reflectance)
PUPIL_MATERIAL_ALBEDO_DEFINE(RoughPlastic,    rough_plastic,    diffuse_reflectance)
PUPIL_MATERIAL_ALBEDO_DEFINE(Hair,            hair,             sigma_a)
#endif
// clang-format on