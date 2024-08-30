SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postHistory AS ph,
  badges AS b,
  users AS u
WHERE u.Id = ph.UserId
  AND u.Id = b.UserId
  AND u.Id = p.OwnerUserId
  AND u.Id = c.UserId
  AND c.Score = 0
  AND p.PostTypeId = 1
  AND p.ViewCount >= 0
  AND p.ViewCount <= 4157
  AND p.FavoriteCount = 0
  AND p.CreationDate <= CAST('2014-09-08 09:58:16' AS timestamp);
