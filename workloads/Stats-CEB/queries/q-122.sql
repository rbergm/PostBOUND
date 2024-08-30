SELECT COUNT(*)
FROM postHistory AS ph,
  posts AS p,
  users AS u,
  badges AS b
WHERE b.UserId = u.Id
  AND p.OwnerUserId = u.Id
  AND ph.UserId = u.Id
  AND ph.CreationDate >= CAST('2010-07-27 18:08:19' AS timestamp)
  AND ph.CreationDate <= CAST('2014-09-10 08:22:43' AS timestamp)
  AND p.PostTypeId = 2;
