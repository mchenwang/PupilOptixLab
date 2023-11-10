#include "mesh.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace Pupil::resource {
    bool ObjMesh::Load(const char* path, ObjMesh& mesh_data, EShapeLoadFlag flags) noexcept {
        uint32_t assimp_flag = aiProcess_Triangulate;
        // if (flags & EShapeLoadFlag::GenNormals) assimp_flag |= aiProcess_GenNormals;
        // if (flags & EShapeLoadFlag::GenSmoothNormals) assimp_flag |= aiProcess_GenSmoothNormals;
        // if (flags & EShapeLoadFlag::GenUVCoords) assimp_flag |= aiProcess_GenUVCoords;
        // if (flags & EShapeLoadFlag::FilpUV) assimp_flag |= aiProcess_FlipUVs;
        // if (flags & EShapeLoadFlag::GenTanget) assimp_flag |= aiProcess_CalcTangentSpace;
        // if (flags & EShapeLoadFlag::OptimizeMesh) assimp_flag |= aiProcess_OptimizeMeshes | aiProcess_OptimizeGraph;

        Assimp::Importer importer;
        const auto       scene = importer.ReadFile(path, assimp_flag);

        if (scene == nullptr) {
            return false;
        }

        if (scene->mNumMeshes != 1) {
            return false;
        }

        uint32_t vertex_index_offset = 0;
        // for (auto i = 0u; i < scene->mNumMeshes; i++) {

        const auto mesh = scene->mMeshes[0];

        bool has_normals   = mesh->HasNormals();
        bool has_texcoords = mesh->HasTextureCoords(0);

        mesh_data.vertex.reserve(mesh->mNumVertices * 3);
        if (has_normals) mesh_data.normal.reserve(mesh->mNumVertices * 3);
        if (has_texcoords) mesh_data.texcoord.reserve(mesh->mNumVertices * 2);

        for (auto j = 0u; j < mesh->mNumVertices; j++) {
            mesh_data.vertex.emplace_back(mesh->mVertices[j].x);
            mesh_data.vertex.emplace_back(mesh->mVertices[j].y);
            mesh_data.vertex.emplace_back(mesh->mVertices[j].z);

            if (has_normals) {
                mesh_data.normal.emplace_back(mesh->mNormals[j].x);
                mesh_data.normal.emplace_back(mesh->mNormals[j].y);
                mesh_data.normal.emplace_back(mesh->mNormals[j].z);
            }

            if (has_texcoords) {
                mesh_data.texcoord.emplace_back(mesh->mTextureCoords[0][j].x);
                mesh_data.texcoord.emplace_back(mesh->mTextureCoords[0][j].y);
            }
        }

        mesh_data.index.reserve(mesh->mNumFaces * 3);
        for (auto j = 0u; j < mesh->mNumFaces; j++) {
            mesh_data.index.emplace_back(mesh->mFaces[j].mIndices[0] + vertex_index_offset);
            mesh_data.index.emplace_back(mesh->mFaces[j].mIndices[1] + vertex_index_offset);
            mesh_data.index.emplace_back(mesh->mFaces[j].mIndices[2] + vertex_index_offset);
        }

        vertex_index_offset += mesh->mNumVertices;
        // }
        return true;
    }
}// namespace Pupil::resource