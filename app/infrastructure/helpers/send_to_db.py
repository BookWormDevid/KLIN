# from app.application.dto import StreamEventDto
#
#
# class SendToDB:
#     async def save_yolo(self, event: StreamEventDto):
#         data = event.payload
#
#         for det in data["detections"]:
#             obj = YoloDetectionModel(
#                 id=event.id,
#                 stream_id=event.stream_id,
#                 camera_id=event.camera_id,
#                 class_id=det["class_id"],
#                 class_name=det["class_name"],
#                 confidence=det["confidence"],
#                 bbox=det["bbox"],
#                 timestamp=det["timestamp"],
#             )
#             self.session.add(obj)
#
#         await self.session.commit()
#
#     async def save_mae(self, event: StreamEventDto):
#         data = event.payload
#
#         obj = MaeDetectionModel(
#             id=event.id,
#             stream_id=event.stream_id,
#             camera_id=event.camera_id,
#             label=data["label"],
#             confidence=data["confidence"],
#             probs=data["probs"],  # JSON поле
#             start_ts=data["start_ts"],
#             end_ts=data["end_ts"],
#         )
#
#         self.session.add(obj)
#         await self.session.commit()
#
#     async def save_x3d(self, event: StreamEventDto):
#         data = event.payload
#
#         obj = X3DDetectionModel(
#             id=event.id,
#             stream_id=event.stream_id,
#             camera_id=event.camera_id,
#             probability=data["prob"],
#             timestamp=data["timestamp"],
#         )
#
#         self.session.add(obj)
#         await self.session.commit()
